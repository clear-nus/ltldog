import sys
import os
from tqdm import tqdm
import time
import pdb

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.dataset.pusht_dataset import PushTLTLValueDataset
from diffusion_policy.model.diffusion.value_classifier import ConditionalValueUnet1D


def test_ConditionalValueUnet1D_model():
    obs_dim = 5
    n_obs_steps = 4
    tic = time.time()
    model = ConditionalValueUnet1D(
        horizon=64,
        input_dim=2,
        propositions=['p0', 'p1'],
        global_cond_dim = obs_dim*n_obs_steps,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8,
        # legacy=False,
    ).to('cuda')
    toc = time.time()
    print(f"Constructing model used {(toc-tic):.3f} secs.\n")

    bsz = 4
    tic = time.time()
    val = model.forward(
        sample = torch.randn(bsz, 64, 2, device='cuda'), 
        timestep = torch.randint(1, 64, (bsz,), device='cuda'), 
        global_cond = torch.randn(bsz, obs_dim*n_obs_steps, device='cuda'),
        formulas=["(G~p0)"]*bsz, 
        # legacy = False,
    )
    toc = time.time()
    print(f"Forward pass used {(toc-tic)*1000} ms.\n")

    print(f"Value: {val}")


if __name__ == '__main__':
    test_ConditionalValueUnet1D_model()
