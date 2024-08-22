from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import logging
from einops import rearrange, reduce
from copy import deepcopy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.guide.ps_guide import PSGuide
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

logger = logging.getLogger(__name__)

class DiffusionUnetLowdimPSPolicy(DiffusionUnetLowdimPolicy):
    def __init__(self, 
                 guide: PSGuide, 
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
            threshold=0.0,
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

        guide_step_cnt = 0
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # 2. predict model output
            model_output = model(
                trajectory, 
                t, 
                local_cond=local_cond, 
                global_cond=global_cond
            )
            # 3. compute previous timestep trj: x_t -> x_t-1
            scheduler_output_t = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            )
            trj_t_1 = scheduler_output_t.prev_sample
            with torch.no_grad():
                if t >= max(1, self.t_stopgrad):
                    # compute the original sample trj_0 from t_2: x_t-1 -> x_0
                    trj_0_from_t_1 = self.predict_x_0(
                        trj_t_1, 
                        t - 1, 
                        local_cond=local_cond, 
                        global_cond=global_cond,
                        scheduler=scheduler, 
                        generator=generator, 
                        **kwargs
                    )
                    unnormed_trj_0_from_t_1 = self.normalizer['action'].unnormalize(trj_0_from_t_1)
                    y = guide(unnormed_trj_0_from_t_1)
                    if y.min() >= threshold:
                        ## no guidance needed in this step
                        trajectory = trj_t_1
                        continue

            # 4. perform gradient ascent for guidance
            with torch.enable_grad():
                ## Get x_0_hat from timestep t: x_t -> x_0_hat
                trj_t = trajectory.detach().requires_grad_()
                trj_0_hat_from_t = self.predict_x_0(
                    trj_t, 
                    t, 
                    local_cond=local_cond, 
                    global_cond=global_cond,
                    scheduler=scheduler, 
                    generator=generator, 
                    **kwargs
                )
                y, grad = guide.gradients(
                    x_prev = trj_t, 
                    x_0_hat = trj_0_hat_from_t, 
                    normalized = True, 
                    custom_normalizer = self.normalizer['action']
                )

                for _ in range(self.n_guide_steps):
                    ## stop guidance near the end
                    if t < self.t_stopgrad:
                        break
                    
                    trj_t_1 = trj_t_1 + self.grad_scale * grad
                    guide_step_cnt += 1
                    ## if the predicted value from t-1 satisfies, stop guidance
                    trj_0_from_t_1 = self.predict_x_0(
                        trj_t_1, 
                        t - 1, 
                        local_cond=local_cond, 
                        global_cond=global_cond,
                        scheduler=scheduler, 
                        generator=generator, 
                        **kwargs
                    )
                    unnormed_trj_0_from_t_1 = self.normalizer['action'].unnormalize(trj_0_from_t_1)
                    y = guide(unnormed_trj_0_from_t_1)
                    if y.min() >= threshold:
                        break
                    grad[y>threshold] = 0
                #end for
            trajectory = trj_t_1
        #end for
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        # logger.info(f"Guided steps: {guide_step_cnt}\n")

        return trajectory

    
    def predict_x_0(self, 
                    x_t, 
                    t,
                    local_cond, global_cond,
                    scheduler: DDPMScheduler, 
                    generator, 
                    **kwargs):
        model = self.model
        model_output_t = model(
            x_t, 
            t, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        x_0 = scheduler.step(
            model_output_t, 
            t, 
            x_t, 
            generator=generator,
            **kwargs
        ).pred_original_sample

        return x_0
