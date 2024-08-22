from collections import namedtuple

import torch
import einops

from .arrays import to_np, to_torch, apply_dict

Trajectories = namedtuple('Trajectories', 'actions observations')

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
        conditions = apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, priors=None, debug=False, batch_size=1):

        conditions = self._format_conditions(conditions, batch_size)
        ## run reverse diffusion process
        sample = self.diffusion_model(conditions) if priors is None else self.diffusion_model(conditions, to_torch(priors))
        sample = to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations)
        return action, trajectories
