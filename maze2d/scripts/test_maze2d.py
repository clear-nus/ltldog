import os
import sys
import json
from os.path import join

import numpy as np
import torch

import diffuser.guides as guides
import diffuser.datasets as datasets
import diffuser.utils as utils

from diffuser.utils.ast_builder import ASTBuilder
from diffuser.utils.ltl_parser import LTLParser

from dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator
sys.setrecursionlimit(10000)
os.environ['TL_RECORD_TRACE'] = '1'

from config.maze2d_constraints import con_groups

def get_graphs(props, ltls):
    str2tup_converter = LTLParser(propositions=props)
    tree_builder = ASTBuilder(propositions=props)
    formula_tups = [str2tup_converter(form_str) for form_str in ltls]
    graphs = np.array([[tree_builder(tup).to('cuda')] for tup in formula_tups])

    ltl_embed_output_dim = 32
    for i in range(graphs.shape[0]):
        d = graphs[i][0].nodes[None].data['feat'].size()
        root_weight = torch.ones((1, ltl_embed_output_dim))
        others_weight = torch.zeros((d[0]-1, ltl_embed_output_dim))
        weight = torch.cat([root_weight, others_weight])
        graphs[i][0].nodes[None].data['is_root'] = weight.cuda()
    return graphs

def get_policy_rg(diffusion, normalizer, args):
    rg_value_experiment = utils.load_diffusion(args.logbase, args.dataset, args.rg_value_loadpath, epoch=args.rg_value_epoch)
    rg_value_function = rg_value_experiment.ema

    guide_config = utils.Config('guides.ValueGuide', model=rg_value_function, graph=None, verbose=False)

    guide = guide_config()
    policy = guides.GuidedPolicy(guide              = guide,
                                 diffusion_model    = diffusion,
                                 normalizer         = normalizer,
                                 ## sampling kwargs
                                 sample_fn          = guides.n_step_guided_p_sample,
                                 n_guide_steps      = args.n_guide_steps.get('rg'),
                                 scale              = args.scale.get('rg'),
                                 t_stopgrad         = args.t_stopgrad.get('rg'),
                                 scale_grad_by_std  = args.scale_grad_by_std.get('rg'),
                                 return_diffusion   = True)
    return policy

def get_policy_dps(diffusion, normalizer, args):
    evaluator = DTL_Cont_Cons_Evaluator(device='cuda')
    evaluator.set_atomic_props(con_groups[args.dataset])

    guide_config = utils.Config('guides.PSGuide', evaluator=evaluator, verbose=False)
    guide = guide_config()
    policy = guides.GuidedPolicy(guide              = guide,
                                 diffusion_model    = diffusion,
                                 normalizer         = normalizer,
                                 ## sampling kwargs
                                 sample_fn          = guides.n_step_ps_p_sample,
                                 n_guide_steps      = args.n_guide_steps.get('dps'),
                                 scale              = args.scale.get('dps'),
                                 t_stopgrad         = args.t_stopgrad.get('dps'),
                                 scale_grad_by_std  = args.scale_grad_by_std.get('dps'),
                                 return_diffusion   = True)
    return policy

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('test')

env = datasets.load_environment(args.dataset)

#---------------------------------- tester ----------------------------------#

props = ['p0','p1','p2','p3','p4','p5']
filename_ltls = 'datasets/ltls_until.txt'
with open(filename_ltls) as file:
    ltls=file.read()
    ltls=ltls.split("\n")[0:-1]
ltls = ltls
n_ltl = len(ltls)
n_train_ltls = int(0.8*n_ltl)
graphs = get_graphs(props, ltls)

tester_dtl = DTL_Cont_Cons_Evaluator(device='cuda')
tester_dtl.set_atomic_props(con_groups[args.dataset])

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policies = {
    'diffuser': guides.Policy(diffusion, dataset.normalizer),
    'rg': get_policy_rg(diffusion, dataset.normalizer, args),
    'dps': get_policy_dps(diffusion, dataset.normalizer, args),
}

#---------------------------------- main loop ----------------------------------#

n_restart = 10

metrics = ['value_gen','satisfaction_gen','value_rollout','satisfaction_rollout','total_r','score']
results = {}
for key, val in policies.items():
    results[key] = {}
    for metric in metrics:
        results[key][metric] = {'all': np.zeros((n_ltl,n_restart)).tolist(),
                                'mean': np.zeros(n_ltl).tolist(),
                                'avg': 0.0, 'avg_train': 0.0, 'avg_test': 0.0,}

for idx in range(n_ltl):
    ltl = ltls[idx]
    tester_dtl.set_ltl_formula(ltl)
    if 'rg' in policies:
        policies['rg'].guide.set_graph(graphs[idx])
    if 'dps' in policies:
        policies['dps'].guide.set_ltl_formula(ltl)
        args.logger.info('[ scripts/test_maze2d ] Set {}-th LTL as: {} -> {}'.format(idx, ltl, policies['dps'].guide.evaluator.ltl))
    for seed in range(n_restart):
        observation = env.reset()
        if env.name == "maze2d-umaze-v1":
            while observation[1] > 2.5:
                observation = env.reset()
        if args.conditional:
            args.logger.info('Resetting target')
            env.set_target()

        target = env._target
        cond = {
            diffusion.horizon - 1: np.array([*target, 0, 0]),
        }

        for key, policy in policies.items():
            env.set_state(observation[0:2], observation[2:4])
            ## observations for rendering
            rollout_obs = [observation.copy()]
            rollout_actions = []

            total_reward = 0
            for t in range(env.max_episode_steps):

                state = env.state_vector().copy()

                if t == 0:
                    cond[0] = observation

                    action, samples, chains, sample = policy(cond, batch_size=args.batch_size)
                    actions = samples.actions[0]
                    sequence = samples.observations[0]

                    # check generated trajectory for a given ltl
                    assignment = tester_dtl.get_assignments(torch.tensor(sample, dtype=torch.float32).cuda())
                    value_gen = tester_dtl.get_evaluations(assignment).cpu().item()
                    results[key]['value_gen']['all'][idx][seed] = value_gen
                    results[key]['satisfaction_gen']['all'][idx][seed] = 1.0 if value_gen > 0 else 0.0

                # next location in path, copy the last location at max timestep of sequence onwards
                if t < len(sequence) - 1:
                    next_waypoint = sequence[t+1]
                else:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0

                ## can use actions or define a simple controller based on state predictions
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

                next_observation, reward, terminal, _ = env.step(action)
                total_reward += reward
                score = env.get_normalized_score(total_reward)

                ## update rollout observations
                rollout_obs.append(next_observation.copy())
                rollout_actions.append(action.copy())

                if terminal:
                    break

            savepath = join(args.savepath, 'sample', f'ltl{idx}_seed{seed}_{key}_plan_rollout.png')
            renderer.composite(savepath, (samples.observations[0], np.array(rollout_obs)), ncol=2)
            np.save('.'.join(savepath.split('.')[:-1] + ['npy']), {'sample':samples.observations[0],
                                                                   'rollout':np.array(rollout_obs)})
        
            # end of rollout
            results[key]['total_r']['all'][idx][seed] = total_reward
            results[key]['score']['all'][idx][seed] = score

            rollout_actions = dataset.normalizer.normalize(np.array(rollout_actions), 'actions')
            rollout_obs = dataset.normalizer.normalize(np.array(rollout_obs[:-1]), 'observations')
            trajectory = np.concatenate([rollout_actions, rollout_obs], axis=1)
            assignment = tester_dtl.get_assignments(torch.tensor(trajectory, dtype=torch.float32).cuda().unsqueeze(dim=0))
            value_rollout = tester_dtl.get_evaluations(assignment).cpu().item()
            results[key]['value_rollout']['all'][idx][seed] = value_rollout
            results[key]['satisfaction_rollout']['all'][idx][seed] = 1.0 if value_rollout > 0 else 0.0
        # end of all methods
        for _method, _ in results.items():
            s = _method + ":\t"
            for _metric in results[_method].keys():
                s += _metric + "(" + str(results[_method][_metric]['all'][idx][seed]) + ")\t"
            args.logger.info(s)
    # end of all seeds
    for _method, _ in results.items():
        s = _method + " mean:\t"
        for _metric in results[_method].keys():
            _mean = np.mean(results[_method][_metric]['all'][idx])
            results[_method][_metric]['mean'][idx] = _mean
            s += _metric + "(" + str(_mean) + ")\t"
        args.logger.info(s)
# end of all ltls
for _method, _ in results.items():
    s = _method + " avg:\t"
    for _metric in results[_method].keys():
        _avg = np.mean(results[_method][_metric]['mean'])
        _avg_train = np.mean(results[_method][_metric]['mean'][0:n_train_ltls])
        _avg_test = np.mean(results[_method][_metric]['mean'][n_train_ltls:n_ltl])
        results[_method][_metric]['avg'] = _avg
        results[_method][_metric]['avg_train'] = _avg_train
        results[_method][_metric]['avg_test'] = _avg_test
        s += _metric + "(" + str(_avg) + ")" + "(" + str(_avg_train) + ")" + "(" + str(_avg_test) + ")\t"
    args.logger.info(s)

## save results as a json file
json_path = join(args.savepath, 'results.json')
json.dump(results, open(json_path, 'w'), indent=2, sort_keys=True)
