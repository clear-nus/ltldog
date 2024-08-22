from typing import Dict
import os
import sys
import torch
import numpy as np
import copy
import pdb
from torch._tensor import Tensor
import zarr
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.simple_util import get_data_stats, \
    normalize_data, create_sample_indices, sample_sequence 

class PushTLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        self.val_mask = val_mask.copy()
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        val_set.val_mask = None
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path,
                 pred_horizon, obs_horizon, action_horizon):

        # read from zarr dataset
        dataset_root = zarr.open(zarr_path, 'r')
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample


class BasePushTValueDataset(PushTLowdimDataset):
    def __init__(self, *args, 
                 constraint_key=None,
                 name_value='', 
                 filename_values = None,
                 normalize=False, 
                 output=print,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output
        self._seed = kwargs.get('seed', 42)

        output(f"[ dataset/pusht_dataset.BasePushTValueDataset ] Created {len(self.sampler)} trajectories of length {self.horizon}.")

        ## Get values from file or calculate them
        if not filename_values:
            filename = 'datasets/' + '-'.join(self.env.name.split('-')[:-1] + ["H"+str(self.horizon), name_value, 'values.npy'])
        else:
            filename = filename_values
        if os.path.isfile(filename):
            self.all_values = np.load(filename)
            self._get_value_bounds(self.all_values)
            self.output(f"[ BasePushTValueDataset.__init__ ] Loaded generated values of \'{name_value}\' constraints (key={constraint_key}) from \'{filename}\'.")
        else:
            self.output(f"[ dataset/pusht_dataset.BasePushTValueDataset ] Cold storage values not found. Generating values of {name_value} constraints (key={constraint_key}) and save values in \'{filename}\'.")
            self.all_values = self._get_values(constraint_key=constraint_key)
            self._get_value_bounds(self.all_values)
            np.save(filename, self.all_values)
            self.output(f"[ dataset/pusht_dataset.BasePushTValueDataset ] Saved generated values of {name_value} constraints to \'{filename}\'")

        self.normed = False
        if normalize:
            self.unormed_all_values = self.all_values.copy()
            self.all_values = self.normalize_value(self.all_values, preserve_sign=True)
            self.normed = True



    def _sample_to_data(self, sample):
        state = sample[self.state_key]
        keypoint = sample[self.obs_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)
        
        data = {
            'obs': obs, # T, D_o
            'state': state, # T, D_s
            'action': sample[self.action_key], # T, D_a
        }

        return data


    def _get_values(self):
        raise NotImplementedError("Method of calculating values is not implemented.")


    def _get_bounds(self):
        self.output('[ datasets/pusht_dataset.BasePushTValueDataset ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        self.output('✓')
        return vmin, vmax
    

    def _get_value_bounds(self, value):
        """
        Args:
            value: (time, feature)
        """
        self.output('[ dataset/pusht_dataset.BasePushTValueDataset ] Getting value dataset bounds...', end=' ', flush=True)
        self.val_min = np.inf * np.ones(value.shape[-1])
        self.val_max = -np.inf * np.ones(value.shape[-1])
        for i in range(value.shape[-1]):
            self.val_min[i] = np.min(value[..., i])
            self.val_max[i] = np.max(value[..., i])
        
        self.output(f"value_min: {self.val_min}; value_max: {self.val_max}")
        self.output('✓')

        return self.val_min, self.val_max


    def normalize_value(self, value, preserve_sign=True):
        module = torch if isinstance(value, torch.Tensor) else np
        if preserve_sign:
            normed = value.copy()
            for i in range(value.shape[-1]):
                neg_vals_mask = value[..., i] < 0
                pos_vals_mask = ~neg_vals_mask
                if self.val_max[i] > 0:
                    normed[pos_vals_mask, i] = (value[pos_vals_mask, i] - 0) / self.val_max[i]
                if self.val_min[i] < 0:
                    normed[neg_vals_mask, i] = (0 - value[neg_vals_mask, i]) / self.val_min[i]
            #end for
            assert module.all(normed >= -1) and module.all(normed <= 1), \
                f"[ BasePushTValueDataset.normalize_value ]: Normalized value out of bounds!"
        
        if not preserve_sign:
            ## [0, 1]
            normed = (value - self.val_min) / (self.val_max - self.val_min)
            ## [-1, 1]
            normed = normed * 2 - 1
            
        return normed
    

class PushTLTLValueDataset(BasePushTValueDataset):
    '''
        adds a value field to the datapoints for training the value function of LTL constraints
    '''
    def __init__(self, 
                 *args, 
                 constraint_key='',
                 value_name='ltl_no', 
                 filename_ltls = 'data/pusht/LTLs/ltls_no.txt', 
                 filename_values = None,
                 num_props=4,
                 ltl_val_ratio=0.0,
                 normalize = False,
                 **kwargs):
        with open(filename_ltls) as file:
            ltls=file.read()
            ltls=ltls.split("\n")[0:-1]

        self.ltls = ltls
        self.ltl_train_ratio = 1 - ltl_val_ratio
        self.n_train_ltls = round(self.ltl_train_ratio * len(ltls))
        self.n_validation_ltls = len(ltls) - self.n_train_ltls
        print(f"[ PushTLTLValueDataset.__init__ ]: Loaded {len(ltls)} LTLs from file \'{filename_ltls}\'.  Training on {self.n_train_ltls} LTLs and validating on {self.n_validation_ltls} LTLs.")
        super().__init__(*args, 
                         constraint_key=constraint_key,
                         name_value=value_name, 
                         filename_values=filename_values, 
                         normalize=normalize,
                         **kwargs)
        self.ltl_graphs = self._get_graphs(num_props=num_props)
        
        ## Construct value training set from all values
        train_inds = self._get_trajectories_mask(
            episode_ends=self.replay_buffer.episode_ends, 
            sequence_length=self.horizon, 
            episode_mask=self.train_mask, 
            pad_before=self.pad_before, 
            pad_after=self.pad_after
        )
        assert train_inds.shape[0] == self.all_values.shape[0], \
            f"[ PushTLTLValueDataset.__init__ ]: Value set indices size ({train_inds.shape[0]}) does not match sampler size ({self.all_values.shape[0]})."
        self.values = self.all_values[train_inds, :self.n_train_ltls]


    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask

        inds = self._get_trajectories_mask(
            episode_ends=self.replay_buffer.episode_ends, 
            sequence_length=self.horizon, 
            episode_mask=self.val_mask, 
            pad_before=self.pad_before, 
            pad_after=self.pad_after
        )
        assert inds.shape[0] == self.all_values.shape[0], \
            f"[ PushTLTLValueDataset.get_validation_dataset ]: Value set indices size ({inds.shape[0]}) does not match value size ({self.all_values.shape[0]})."

        if self.n_validation_ltls > 0:
            ## Validation using unseen LTLs and unseen trajectories
            val_set.ltls = self.ltls[self.n_train_ltls:]
            val_set.values = self.all_values[inds, self.n_train_ltls:]
            val_set.ltl_graphs = self.ltl_graphs[self.n_train_ltls:]
            val_set.n_train_ltls = self.n_validation_ltls
        elif np.sum(val_set.train_mask) > 0:
            ## Take all training LTLs with unseen trajectories for validation
            val_set.values = self.all_values[inds, :]
        else:
            ## Validation set is empty
            val_set.values = np.empty((0, self.all_values.shape[1]))
        #end if
        assert val_set.values.shape[0] == len(val_set.sampler), \
            f"[ PushTLTLValueDataset.get_validation_dataset ]: Value set size ({val_set.values.shape[0]}) does not match sampler size ({len(val_set.sampler)})."

        val_set.n_validation_ltls = None
        
        return val_set
    

    def _get_trajectories_mask(self, 
        episode_ends:np.ndarray, sequence_length:int, 
        episode_mask: np.ndarray,
        pad_before: int=0, pad_after: int=0,
        debug:bool=True) -> np.ndarray:
        episode_mask.shape == episode_ends.shape        
        pad_before = min(max(pad_before, 0), sequence_length-1)
        pad_after = min(max(pad_after, 0), sequence_length-1)

        indices = list()
        for i in range(len(episode_ends)):
            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i-1]
            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx
            
            min_start = -pad_before
            max_start = episode_length - sequence_length + pad_after
            
            # range stops one idx before end
            indices += [episode_mask[i]] * (max_start + 1 - min_start)
        indices = np.array(indices, dtype=bool)
        return indices


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx_trj = idx // self.n_train_ltls
        idx_ltl = idx % self.n_train_ltls

        batch = super().__getitem__(idx_trj)

        value = self.values[idx_trj, idx_ltl]
        value = torch.tensor([value], dtype=torch.float32)


        ltl_idx = torch.tensor(idx_ltl, dtype=torch.long)

        ltlvalue_batch = batch | {'value': value, 'ltl_idx': ltl_idx} 
        
        return ltlvalue_batch



    def __len__(self) -> int:
        return len(self.sampler) * self.n_train_ltls


    def _get_values(self, constraint_key, batch_size=32768*4):
        from diffusion_policy.dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator
        from diffusion_policy.constraints.pusht_constraints import constraint_dict
        # Raise the recursion limit to avoid problems when parsing formulas
        sys.setrecursionlimit(10000)
        # Setting TL_RECORD_TRACE asks DTL to record the evaluation trace.
        # Using this we can find the conflicting part between logits and formula.
        os.environ['TL_RECORD_TRACE'] = '1'

        props = ['p0', 'p1', 'p2', 'p3']
        evaluator = DTL_Cont_Cons_Evaluator(device='cuda')
        # assignments = self._get_assignments(evaluator, batch_size=batch_size)
        # sampler = BatchSampler(SequentialSampler(range(len(assignments))), batch_size=batch_size, drop_last=False)
        obj_super = super()
        class IterSuper(torch.utils.data.Dataset):
            def __init__(self, obj_super):
                super().__init__()
                self.obj_super = obj_super
            def __len__(self):
                return self.obj_super.__len__()
            def __getitem__(self, idx):
                return self.obj_super.__getitem__(idx)
        
        iter_super_train = IterSuper(obj_super)
        loader_super = torch.utils.data.DataLoader(iter_super_train, batch_size=batch_size)
        

        values_train = np.full(
            ( len(self.sampler), self.n_train_ltls ), 
            fill_value=np.nan,
            dtype=np.float64
        )
        for idx_ltl in tqdm(range(self.n_train_ltls)):
            ltl = self.ltls[idx_ltl]
            ltl = ltl.replace('False', 'FALSE')
            ltl = ltl.replace('True', 'TRUE')
            evaluator.set_props_cons_and_ltl(constraint_dict[constraint_key], props, ltl)
            self.output(f'[ datasets/pusht_dataset ] get values for {idx_ltl}-th LTL: {ltl} -> {evaluator.ltl}')
            value = []

            for batch in tqdm(loader_super):
                # trjs = torch.tensor(batch.trajectories, dtype=torch.float32).cuda()
                trjs = batch['state'].cuda()
                # normed_trjs = normalizer.normalize(trjs)
                # assignment = evaluator.get_assignments(normed_trjs)
                assignment = evaluator.get_assignments(trjs)
                # normed_val = self.normalize_value(evaluator.get_evaluations(assignment).cpu())
                # value.append(normed_val)
                value.append(evaluator.get_evaluations(assignment).cpu())
            values_train[:, idx_ltl] = torch.cat(value).numpy()
        #end for

        return values_train


    def _get_graphs(self, num_props=4, prop_names=None):
        from diffusion_policy.common.ast_builder import ASTBuilder
        from diffusion_policy.common.ltl_parser import LTLParser 
        
        if prop_names is None:
            assert num_props is not None, \
                f"[ PushTLTLValueDataset._get_graphs ]: num_props must be specified if prop_names is not given."
            # props = ['p0','p1','p2','p3']
            props = [f"p{i}" for i in range(num_props)]
        else:
            assert len(prop_names) == num_props, \
                f"[ PushTLTLValueDataset._get_graphs ]: Length of input proposition list (prop_names[{len(prop_names)}]) does not match the declared number (num_props={num_props})."
        str2tup_converter = LTLParser(propositions=props)
        tree_builder = ASTBuilder(propositions=props)

        formula_tups = [str2tup_converter(form_str) for form_str in self.ltls]
        graphs = np.array([[tree_builder(tup).to('cuda')] for tup in formula_tups])

        ltl_embed_output_dim = 32
        for i in range(graphs.shape[0]):
            d = graphs[i][0].nodes[None].data['feat'].size()
            root_weight = torch.ones((1, ltl_embed_output_dim))
            others_weight = torch.zeros((d[0]-1, ltl_embed_output_dim))
            weight = torch.cat([root_weight, others_weight])
            graphs[i][0].nodes[None].data['is_root'] = weight.cuda()
        # end for
        
        return graphs
