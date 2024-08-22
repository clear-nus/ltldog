import pdb
import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if y > 0:
            break

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

# """
def n_step_ps_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    threshold = 0.0
    with torch.no_grad():
        x_recon = model.predict_start_from_noise(x, t=t, noise=model.model(x, cond, t))
        if model.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        model_mean, _, model_log_variance = model.q_posterior(x_start=x_recon, x_t=x, t=t)

        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)

        # no noise when t == 0
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        x_t = model_mean + model_std * noise
        x_t = apply_conditioning(x_t, cond, model.action_dim)

    y = guide(model.model(x_t, cond, t))
    if y > threshold:
        return x_t.detach_(), y

    with torch.enable_grad():
        x_prev = x.detach_().requires_grad_()
        x_0_hat = model.predict_start_from_noise(x_prev, t=t, noise=model.model(x_prev, cond, t))
        if model.clip_denoised:
            x_0_hat.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        x_0_hat = apply_conditioning(x_0_hat, cond, model.action_dim)

        y, grad = guide.gradients(x_prev, x_0_hat)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0
        
        for _ in range(n_guide_steps):
            x_t = x_t + scale * grad
            x_t = apply_conditioning(x_t, cond, model.action_dim)

            y = guide(model.model(x_t, cond, t))
            if y > threshold:
                break

    return x_t.detach_(), y
