from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.value_classifier import ConditionalValueUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionUnetLTLValue(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalValueUnet1D,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            noise_scheduler: DDPMScheduler,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model.to(device)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.noise_scheduler = noise_scheduler
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
    
    
    # ========= inference  ============
    def predict_value(self, x, cond, t, formulas):
        val = self.model(
            sample = x, 
            timestep = t, 
            global_cond = cond,
            formulas=formulas 
        )
        return val


    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss(self, batch, formula, **kwargs):
        # normalize input
        assert 'valid_mask' not in batch
        ## Normalize issues with value and formula:
        ## Formulas don't need to be normalized
        ## Take the value out first, don't normalize it for now
        target = batch['value']
        nbatch = self.normalizer.normalize({key: batch[key] for key in ['obs', 'action']})
        obs = nbatch['obs']
        action = nbatch['action']
        # formula =nbatch['formula']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)


        # raise NotImplementedError

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the value from a noisy trajectory
        pred = self.model(
            sample = noisy_trajectory, 
            timestep = timesteps, 
            formulas = formula,
            local_cond=local_cond, 
            global_cond=global_cond
        )

        # pred_type = self.noise_scheduler.config.prediction_type 
        # if pred_type == 'epsilon':
        #     target = noise
        # elif pred_type == 'sample':
        #     target = trajectory
        # else:
        #     raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
