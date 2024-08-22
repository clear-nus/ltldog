import sys
import os
from tqdm import tqdm
import pdb

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.dataset.pusht_dataset import PushTLTLValueDataset


def test_LTLValueDataset():
    path = "data/pusht/pusht_cchi_v7_replay.zarr"
    augmented_pusht_path = "data/pusht/merged_pusht_cchi_v7_replay-seed_42r_33rf.zarr"
    # VALUE_DATA_FILE_NAME = "data/pusht/pusht-H16-ef_ltls_no_4-values-merged-seed_42r_33rf.npy"
    VALUE_DATA_FILE_NAME = "data/pusht/pusht-H192O2-ef_ltls_no_4-values-merged-seed_42r_33rf.npy"

    ## H192-O2-A192
    horizon = 192
    pad_before = 2 - 1
    pad_after = 192 - 1
    
    # ## H16-O2-A8
    # horizon = 16
    # pad_before = 2 - 1
    # pad_after = 8 - 1
    
    dataset = PushTLTLValueDataset(
        # path,
        augmented_pusht_path,
        horizon, 
        pad_before, 
        pad_after, 
        filename_ltls='data/pusht/LTLs/ltls_no_4.txt',
        value_name="ef_ltls_no_4",
        filename_values=VALUE_DATA_FILE_NAME,
        val_ratio=0.3,
        normalize=False,
    ) 
    loader = DataLoader(dataset, batch_size=8192, shuffle=True, num_workers=0, pin_memory=True, pin_memory_device='cuda:0')
    normalizer = dataset.get_normalizer()
    print(f"Training set size: {len(dataset.sampler)}")

    print(f"min: {dataset.val_min}, max: {dataset.val_max}")

    validation_set = dataset.get_validation_dataset()
    if len(validation_set) > 0:
        val_loader = DataLoader(validation_set, batch_size=8192, shuffle=True, num_workers=0, pin_memory=True, pin_memory_device='cuda:0')
        print(f"Validation set size: {len(validation_set.sampler)}")

    for batch in tqdm(loader):
        ltl_indices = batch["ltl_idx"]
        batch.update({'ltl': dataset.ltl_graphs[ltl_indices.cpu().numpy()]})
        for key, val in batch.items():
            continue
            # print(key, val.shape)
            # if key == "ltl" or key == "ltl_idx":
            #     print(val[0])
            # else:
            #     print(val[0, :3])
    
    if len(validation_set) > 0:
        for batch in tqdm(val_loader):
            ltl_indices = batch["ltl_idx"]
            batch.update({'ltl': validation_set.ltl_graphs[ltl_indices.cpu().numpy()]})


    print("No errors so far. Test passed.")


if __name__ == '__main__':
    test_LTLValueDataset()
