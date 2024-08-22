import os
import copy
import json
from tqdm import tqdm

import numpy as np
import torch
import einops

from diffuser.datasets import load_environment

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from .policies import Policy

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
        output=print,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.output = output
        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                if hasattr(self.dataset, 'graphs'):
                    batch = batch._replace(ltls = self.dataset.graphs[batch.ltls.cpu().numpy()])

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                self.output(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            self.step += 1

    def test(self, epoch, bs=1, n_avg=5, vis_freq=10):
        self.output("[ utils/training ] Testing...")

        env = load_environment(self.dataset.env.name)
        policy = Policy(self.ema_model, self.dataset.normalizer)

        #---------------------------------- main loop ----------------------------------#
        all_score, all_t, all_R, all_d = np.zeros(n_avg), np.zeros(n_avg), np.zeros(n_avg), np.zeros(n_avg)
        for idx in range(n_avg):

            observation = env.reset()

            ## set conditioning xy position to be the goal
            target = env._target
            cond = {
                self.ema_model.horizon - 1: np.array([*target, 0, 0]),
            }

            ## observations for rendering
            rollout = [observation.copy()]

            total_reward = 0
            for t in range(env.max_episode_steps):

                state = env.state_vector().copy()

                ## can replan if desired, but the open-loop plans are good enough for maze2d
                ## that we really only need to plan once
                if t == 0:
                    cond[0] = observation

                    action, samples = policy(cond, batch_size=bs)
                    actions = samples.actions[0]
                    sequence = samples.observations[0]

                # ####
                if t < len(sequence) - 1:
                    next_waypoint = sequence[t+1]
                else:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0

                ## can use actions or define a simple controller based on state predictions
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                ####

                next_observation, reward, terminal, _ = env.step(action)
                total_reward += reward
                score = env.get_normalized_score(total_reward)

                ## update rollout observations
                rollout.append(next_observation.copy())

                if terminal:
                    break

                observation = next_observation

            # save plan and rollout
            savepath = os.path.join(self.logdir, 'test', f'{epoch}_{idx}_plan_rollout.png')
            self.renderer.composite(savepath, (samples.observations[0], np.array(rollout)[:-1]), ncol=2)

            # logger.finish(t, env.max_episode_steps, score=score, value=0)
            all_score[idx] = score
            all_t[idx] = t
            all_R[idx] = total_reward
            all_d[idx] = terminal

        output_str = f'[ utils/training ] Test R avg: {all_R.mean():.2f} | score avg: {all_score.mean():.4f} | {action} | '
        if 'maze2d' in self.dataset.env.name:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            output_str += f'maze | pos_final: {xy} | goal: {goal}'
        self.output(output_str)

        ## save result as a json file
        json_path = os.path.join(self.logdir, 'rollout.json')
        json_data = {'epoch': epoch, 'score': all_score.tolist(), 'step': all_t.tolist(), 'return': all_R.tolist(), 'term': all_d.tolist()}
        json.dump(json_data, open(json_path, 'a'), indent=2, sort_keys=True)

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        self.output(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''
        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, 'sample', f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):
            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## get trajectories and condition at t=0 from batch
            trajectories = to_np(batch.trajectories)

            ## [ batch_size x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]
            refs = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            ## [ n_samples x horizon x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, 'sample', f'{self.step}-{i}-ref-sample.png')
            self.renderer.composite(savepath, np.concatenate([refs, observations]), ncol=5)
