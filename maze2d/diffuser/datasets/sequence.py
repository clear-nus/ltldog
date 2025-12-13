import os
import sys

from collections import namedtuple
import random
import numpy as np
import torch
from tqdm import tqdm
import gym

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
LTLValueBatch = namedtuple('LTLValueBatch', 'trajectories conditions values ltls')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, 
        output=print, **kwargs):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn, output=output, **kwargs)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize(output)

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        output(fields)
        self.output = output
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ConstraintValueDataset(GoalDataset):
    '''
        adds a value field to the datapoints for training the value function of constraints
    '''
    def __init__(self, *args, normed=False, name_value='', **kwargs):
        super().__init__(*args, **kwargs)

        filename = 'datasets/' + '-'.join(self.env.name.split('-')[:-1] + ["H"+str(self.horizon), name_value, 'values.npy'])
        if os.path.isfile(filename):
            self.values = np.load(filename)
            self.output(f'[ datasets/sequence ] Load generated values of {name_value} constraints from {filename}')
            assert len(self.values) == len(self.indices)
        else:
            self.values = self._get_values()
            assert len(self.values) == len(self.indices)
            np.save(filename, self.values)
            self.output(f'[ datasets/sequence ] Save generated values of {name_value} constraints to {filename}')

        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_values(self):
        raise NotImplementedError("Method of calculating values is not implemented.")

    def _get_bounds(self):
        self.output('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        self.output('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

class LTLsValueDataset(ConstraintValueDataset):
    '''
        adds ltls and values field to the datapoints for training the value function of different LTL constraints
    '''
    def __init__(self,
                 *args,
                 filename_ltls='datasets/ltls_until.txt',
                 props=['p0','p1','p2','p3','p4','p5'],
                 **kwargs):
        with open(filename_ltls) as file:
            ltls=file.read()
            ltls=ltls.split("\n")[0:-1]
        self.ltls = ltls
        self.n_train_ltls = int(0.8*len(ltls))
        self.n_test_ltls = len(ltls) - self.n_train_ltls
        self.props = props
        self.graphs = self._get_graphs()
        super().__init__(*args, name_value='ltls-until', **kwargs)

    def _get_graphs(self):
        from diffuser.utils.ast_builder import ASTBuilder
        from diffuser.utils.ltl_parser import LTLParser, LTLParseError
        str2tup_converter = LTLParser(propositions=self.props)
        tree_builder = ASTBuilder(propositions=self.props)

        formula_tups = [str2tup_converter(form_str) for form_str in self.ltls]
        graphs = np.array([[tree_builder(tup).to('cuda')] for tup in formula_tups])

        ltl_embed_output_dim = 32
        for i in range(graphs.shape[0]):
            d = graphs[i][0].nodes[None].data['feat'].size()
            root_weight = torch.ones((1, ltl_embed_output_dim))
            others_weight = torch.zeros((d[0]-1, ltl_embed_output_dim))
            weight = torch.cat([root_weight, others_weight])
            graphs[i][0].nodes[None].data['is_root'] = weight.cuda()
        return graphs

    def _get_values(self):
        from dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator
        from config.maze2d_constraints import con_groups
        sys.setrecursionlimit(10000)
        os.environ['TL_RECORD_TRACE'] = '1'

        evaluator = DTL_Cont_Cons_Evaluator(device='cuda')
        evaluator.set_atomic_props(con_groups[self.env.name])
        batch_size = 512
        obj_super = super()
        class IterSuper(torch.utils.data.Dataset):
            def __len__(self):
                return obj_super.__len__()
            def __getitem__(self, idx):
                return obj_super.__getitem__(idx)
        iter_super = IterSuper()
        loader_super = torch.utils.data.DataLoader(iter_super, batch_size=batch_size)

        values = np.ones([len(self.indices),len(self.ltls)])
        for idx_ltl in tqdm(range(len(self.ltls))):
            ltl = self.ltls[idx_ltl]
            evaluator.set_ltl_formula(ltl)
            self.output(f'[ datasets/sequence ] get values for {idx_ltl}-th LTL: {ltl} -> {evaluator.ltl}')
            value = []
            for batch in tqdm(loader_super):
                trjs = torch.tensor(batch.trajectories, dtype=torch.float32).cuda()
                assignment = evaluator.get_assignments(trjs)
                value.append(evaluator.get_evaluations(assignment).cpu())
            values[:,idx_ltl] = torch.cat(value).numpy()

        return values
    
    def __len__(self):
        return len(self.indices)*self.n_train_ltls

    def __getitem__(self, idx):
        idx_trj = idx // self.n_train_ltls
        idx_ltl = idx % self.n_train_ltls

        batch = super().__getitem__(idx_trj)

        value = self.values[idx_trj, idx_ltl]
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)

        ltl = np.array(idx_ltl)

        ltlvalue_batch = LTLValueBatch(*batch, value, ltl)
        return ltlvalue_batch

    def test__len__(self):
        return len(self.indices)*self.n_test_ltls

    def test__getitem__(self, idx):
        idx_trj = idx // self.n_test_ltls
        idx_ltl = idx % self.n_test_ltls + self.n_train_ltls
        batch = super().__getitem__(idx_trj)

        value = self.values[idx_trj, idx_ltl]
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        # value = np.array([value>0], dtype=np.float32)
        
        # ltl = self.ltls[idx_ltl]
        ltl = np.array(idx_ltl)

        ltlvalue_batch = LTLValueBatch(*batch, value, ltl)
        return ltlvalue_batch
