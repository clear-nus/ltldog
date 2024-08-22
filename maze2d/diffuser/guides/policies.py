from collections import namedtuple
# import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations')
GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations values')

class Policy:

    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):

        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        sample, chains = self.diffusion_model(conditions, return_diffusion=True)
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        if 0 != self.action_dim:
            actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        chains = self.normalizer.unnormalize(utils.to_np(chains[:,:,:,self.action_dim:]), 'observations')

        trajectories = Trajectories(actions, observations)
        return action, trajectories, chains, sample


class GuidedPolicy(Policy):

    def __init__(self, guide, diffusion_model, normalizer, **sample_kwargs):
        super().__init__(diffusion_model, normalizer)

        self.guide = guide
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, debug=False, batch_size=1, verbose=True):

        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        sample = samples.trajectories
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        if 0 != self.action_dim:
            actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        chains = self.normalizer.unnormalize(utils.to_np(samples.chains[:,:,:,self.action_dim:]), 'observations')

        trajectories = GuidedTrajectories(actions, observations, samples.values)
        return action, trajectories, chains, sample
