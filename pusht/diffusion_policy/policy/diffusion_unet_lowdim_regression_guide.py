from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.guide.value_guide import ValueGuide
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionUnetLowdimRGPolicy(DiffusionUnetLowdimPolicy):
    def __init__(self, 
                 guide: ValueGuide, 
                 n_guide_steps=2, 
                 grad_scale=1.,
                 t_stopgrad = 2, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.guide = guide
        self.n_guide_steps = n_guide_steps
        self.grad_scale = grad_scale
        self.t_stopgrad = t_stopgrad
    
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            guide=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        if guide is None:
            guide = self.guide

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 1.1 perform gradient ascent for guidance
            for _ in range(self.n_guide_steps):
                ## stop guidance near the end
                if t < self.t_stopgrad:
                    break
                
                with torch.enable_grad():
                    y, grad = guide.gradients(
                        trajectory, 
                        t, 
                        local_cond=local_cond, 
                        global_cond=global_cond
                    )
                ## if the predicted value satisfies, stop guidance
                if y.min()>=0:
                    break
                grad[y>0] = 0
                trajectory = trajectory + self.grad_scale * grad
            #end for
            # 2. predict model output
            model_output = model(
                trajectory, 
                t, 
                local_cond=local_cond, 
                global_cond=global_cond
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

