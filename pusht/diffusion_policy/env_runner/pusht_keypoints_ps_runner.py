import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import logging
import pdb
import wandb.sdk.data_types.video as wv
from typing import List
from copy import deepcopy

from diffusion_policy.policy.diffusion_unet_lowdim_postsamp_guide import DiffusionUnetLowdimPSPolicy
from diffusion_policy.dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.pusht_keypoints_rg_runner import PushTKeypointsRGRunner

logger = logging.getLogger(__name__)

class PushTKeypointsPSRunner(PushTKeypointsRGRunner):
    def __init__(self,
            output_dir,
            constraint_key, 
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=1.0,
            n_envs=None
        ):
        super().__init__(output_dir, constraint_key, keypoint_visible_rate, 
                         n_train, n_train_vis, train_start_seed, n_test, 
                         n_test_vis, legacy_test, test_start_seed, max_steps, 
                         n_obs_steps, n_action_steps, n_latency_steps, fps, 
                         crf, agent_keypoints, past_action, 
                         tqdm_interval_sec, n_envs)

    
    def run(self, 
            policy: DiffusionUnetLowdimPSPolicy, 
            ltl_formulas: list,
            tester_dtl: DTL_Cont_Cons_Evaluator):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_ltls = len(ltl_formulas)
        n_chunks = math.ceil(n_inits / n_envs)

        ## allocate data
        all_video_paths = [[None]* n_inits ] * n_ltls
        all_rewards = np.empty((n_inits, self.max_steps, n_ltls))
        all_satisfaction = np.empty((n_inits, n_ltls))
        all_state_trjs = np.empty((n_inits, self.max_steps+1, 5, n_ltls))
        # all_video_paths = [None] * n_inits 
        # all_rewards = [None] * n_inits 
        # all_satisfaction = [None] * n_inits 

        ## Fill with NaN to protect missing data
        all_rewards.fill(np.nan)
        all_satisfaction.fill(np.nan)
        all_state_trjs.fill(np.nan)

        logger.info(f"Running evaluation for {n_inits} envs, {n_ltls} LTLs, {n_chunks} chunks")
        for ltl_idx in tqdm.tqdm(range(n_ltls)):
            tester_dtl.set_ltl_only(ltl_formulas[ltl_idx])
            assert ltl_formulas is not None and len(ltl_formulas) > 0
            policy.guide.set_ltl_formula(ltl_formulas[ltl_idx])
            for chunk_idx in range(n_chunks):
                start = chunk_idx * n_envs
                end = min(n_inits, start + n_envs)
                this_global_slice = slice(start, end)
                this_n_active_envs = end - start
                this_local_slice = slice(0,this_n_active_envs)
                
                this_init_fns = self.env_init_fn_dills[this_global_slice]
                n_diff = n_envs - len(this_init_fns)
                if n_diff > 0:
                    this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
                assert len(this_init_fns) == n_envs

                # init envs
                env.call_each(
                    'run_dill_function', 
                    args_list=[(x,) for x in this_init_fns],
                    kwargs_list=[{'prefix': f"ltl{ltl_idx:03d}_"}] * n_envs,
                )
                env.call('set_ltl_str', ltl_formulas[ltl_idx])

                # start rollout
                obs = env.reset()
                past_action = None
                policy.reset()

                # states = None
                pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsPSRunner {chunk_idx+1}/{n_chunks}", 
                    leave=False, mininterval=self.tqdm_interval_sec)
                done = False
                while not done:
                    Do = obs.shape[-1] // 2
                    # create obs dict
                    np_obs_dict = {
                        # handle n_latency_steps by discarding the last n_latency_steps
                        'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                        'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                    }
                    if self.past_action and (past_action is not None):
                        # TODO: not tested
                        np_obs_dict['past_action'] = past_action[
                            :,-(self.n_obs_steps-1):].astype(np.float32)
                    
                    # device transfer
                    obs_dict = dict_apply(np_obs_dict, 
                        lambda x: torch.from_numpy(x).to(
                            device=device))

                    # run policy
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())

                    # handle latency_steps, we discard the first n_latency_steps actions
                    # to simulate latency
                    action = np_action_dict['action'][:,self.n_latency_steps:]

                    # step env
                    obs, reward, done, info = env.step(action)
                    done = np.all(done)
                    past_action = action
                    
                    ## update pbar
                    pbar.update(action.shape[1])
                pbar.close()

                ## collect data for this round
                # all_video_paths[this_global_slice] = env.render()[this_local_slice]
                ### Prefix the filenames, so that we can distinguish different LTLs
                ### Do it the dump way 
                this_video_paths = env.render()[this_local_slice]
                all_video_paths[ltl_idx][this_global_slice] = self.prefix_filenames(
                    file_paths = this_video_paths, 
                    # prefixes   = [f'ltl{ltl_idx:03}_'] * this_n_active_envs
                    prefixes   = [''] * len(this_video_paths)
                )
                
                # all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
                reward_trjs = env.call('get_attr', 'reward')[this_local_slice]
                reward_trjs = self.pad(reward_trjs, self.max_steps)
                # reward_trjs = np.concatenate(reward_trjs, axis=-1)
                reward_trjs = np.stack(reward_trjs, axis=0)
                all_rewards[this_global_slice, :, ltl_idx] = reward_trjs

                ## evaluate this chunk's value
                # import pdb; pdb.set_trace()
                state_trjs = deepcopy(env.call('get_attr', 'state_trj')[this_local_slice])
                state_trjs = self.pad(state_trjs, self.max_steps+1)  # (bs, max_steps+1, state_dim)
                state_trjs = np.stack(state_trjs, axis=0)   # (bs, max_steps+1, state_dim)
                all_state_trjs[this_global_slice, :, :, ltl_idx] = state_trjs
                values_chunk = self.calc_value(state_trjs, tester_dtl)
                # values_chunk[values_chunk<0] = 0
                # values_chunk[values_chunk>=0] = 1
                all_satisfaction[this_global_slice, ltl_idx] = \
                    (values_chunk>=0).to(dtype=torch.float32).cpu().numpy()[:]
        # import pdb; pdb.set_trace()

        # log
        max_rewards = collections.defaultdict(list)
        mean_satisfactions = collections.defaultdict(list)
        log_data = dict()
        
        log_data['all_rewards'] = all_rewards
        log_data['all_satisfaction'] = all_satisfaction
        log_data['all_state_trjs'] = all_state_trjs
        
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            
            # max_reward = np.max(all_rewards[i])
            ## Should take the average max rewards in a single trj across different LTLs
            rewards_per_ltl = np.max(all_rewards[i], axis=0)
            assert rewards_per_ltl.shape[0] == n_ltls, \
                f"rewards_per_ltl.shape[0]: {rewards_per_ltl.shape[0]}, not equal to n_ltls: {n_ltls}"
            avg_max_reward = np.mean(rewards_per_ltl)
            max_rewards[prefix].append(avg_max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = avg_max_reward

            mean_satisfaction = np.mean(all_satisfaction[i])
            # max_satisfaction = np.max(all_satisfaction[i])
            mean_satisfactions[prefix].append(mean_satisfaction)
            log_data[prefix+f'sim_mean_satisfaction_{seed}'] = mean_satisfaction

            ## visualize sim
            for j in range(n_ltls):
                video_path = all_video_paths[j][i]
                if video_path is not None:
                    sim_video = wandb.Video(video_path)
                    log_data[prefix+f'LTL_{j}/'+f'sim_video_{seed}'] = sim_video
            # # visualize sim
            # video_path = all_video_paths[i]
            # if video_path is not None:
            #     sim_video = wandb.Video(video_path)
            #     log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            avg = prefix+'mean_score'
            avg_value = np.mean(value)
            log_data[avg] = avg_value
            
            std = prefix+'score_std'
            std_value = np.std(value)
            log_data[std] = std_value

            if hasattr(policy, "n_guide_steps") and hasattr(policy, "grad_scale"):
                logger.info(
                    f"stp{policy.n_guide_steps}-scl{policy.grad_scale}-t{n_inits}"\
                    f"--mScr: {avg_value:.4g} ± {std_value:.4g}"
                )
            else:
                logger.info(
                    f"t{n_inits}"\
                    f"--mScr: {avg_value:.4g} ± {std_value:.4g}"
                )

        for prefix, value in mean_satisfactions.items():
            avg = prefix+'mean_satisfaction_rate'
            avg_value = np.mean(value)
            log_data[avg] = avg_value
            
            std = prefix+'satisfaction_rate_std'
            std_value = np.std(value)
            log_data[std] = std_value

            if hasattr(policy, "n_guide_steps") and hasattr(policy, "grad_scale"):
                logger.info(
                    f"stp{policy.n_guide_steps}-scl{policy.grad_scale}-t{n_inits}"\
                    f"--mSR: {avg_value:.4g} ± {std_value:.4g}"
                )
            else:
                logger.info(
                    f"t{n_inits}"\
                    f"--mSR: {avg_value:.4g} ± {std_value:.4g}"
                )

        return log_data

