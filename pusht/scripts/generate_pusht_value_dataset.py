import sys
import os
from tqdm import tqdm
import pdb

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# ROOT_DIR = os.path.dirname(__file__)
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.dataset.pusht_dataset import PushTLTLValueDataset



## Augmented pusht
### EF
#### H16-O2-A8
VALUE_DATA_FILE_NAME = "data/pusht/pusht-H16-ef_ltls_no_4-values-merged-seed_42r_33rf.npy"
#### H192-O2
# VALUE_DATA_FILE_NAME = "data/pusht/pusht-H192O2-ef_ltls_no_4-values-merged-seed_42r_33rf.npy"
# VALUE_DATA_FILE_NAME = "data/pusht/pusht-H192O2-ef_ltls_until_4-values-merged-seed_42r_33rf.npy"


def generate_LTLValueDataset():
    # pusht_path = "data/pusht/pusht_cchi_v7_replay.zarr"
    augmented_pusht_path = "data/pusht/merged_pusht_cchi_v7_replay-seed_42r_33rf.zarr"
    
    # ## H192-O2-A192
    # horizon = 192
    # n_obs_steps = 2
    # n_action_steps = 192
    
    ## H16-O2-A8
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8

    ## Common
    pad_before = n_obs_steps - 1
    pad_after = n_action_steps - 1

    dataset = PushTLTLValueDataset(
        # pusht_path, 
        augmented_pusht_path,
        horizon, 
        pad_before, 
        pad_after, 
        constraint_key="ef_oa",
        filename_ltls='data/pusht/LTLs/ltls_no_4.txt',
        value_name="ef_ltls_no_4",
        filename_values=VALUE_DATA_FILE_NAME,
        normalize=False,
        val_ratio=0.0,
        ltl_val_ratio=0.0,
    ) 

    print(f"min: {dataset.val_min}, max: {dataset.val_max}")

    print(f"Dataset size: {len(dataset.sampler)}")

    loader = DataLoader(dataset, batch_size=8192, shuffle=True, num_workers=0)
    normalizer = dataset.get_normalizer()


    ## calculate graphs for `cnt` batches and check the speed. 
    cnt = 50
    total = cnt
    import time; tic = time.time()
    for batch in tqdm(loader):
        ltl_indices = batch["ltl_idx"]
        batch.update({'ltl': dataset.ltl_graphs[ltl_indices.cpu().numpy()]})
        cnt -= 1
        if cnt <= 0:
            break
    #end for
    toc = time.time()
    total = min(total, total-cnt)
    print(f"Time per batch: {(toc-tic)*1000/total:.1f} ms")

    # import pdb; pdb.set_trace()


def raw_metrics(filename):
    if os.path.isfile(filename):
        values = np.load(filename)
    else:
        raise RuntimeError(f'File {filename} not found!')
    
    n_ltls = values.shape[1]
    n_trjs = values.shape[0]
    
    mask = values>=0 
    n_satisfied = np.sum(mask, axis=0)
    satisfaction_rate = n_satisfied / n_trjs
    
    for i in range(n_ltls):
        print(f"{i}: {n_satisfied[i]}/{n_trjs} = {satisfaction_rate[i]:.2%}")

    print(f"Average satisfaction rate: {satisfaction_rate.mean(): .2%}")


if __name__ == '__main__':
    generate_LTLValueDataset()
    raw_metrics(VALUE_DATA_FILE_NAME)
