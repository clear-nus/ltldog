import os
import sys
import json
import logging
import click
import pdb
import numpy as np
from tqdm import tqdm
from itertools import product

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# # use line-buffering for both stdout and stderr
# sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
# sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

BASE_DIR = str(os.path.dirname(os.path.abspath(__file__)))
GUIDERS = ['baseline', 'ps', 'rg']


## Parameters
num_ltl = 36
trial = 50

ltl_dir = 'data/pusht/LTLs/ltls_until_4.txt'
ltl_train_dir = 'data/pusht/LTLs/ltls_until_4_train.txt'
ltl_test_dir = 'data/pusht/LTLs/ltls_until_4_test.txt'
ltl_debug_dir = 'data/pusht/LTLs/ltls_until_4_debug.txt'
# ltl_dir = ltl_test_dir
# ltl_dir = ltl_debug_dir

max_steps = 191
n_envs = None
# ## PS 
# constraint_type = 'ef_tp'
# steps = [2]
# scales = [1.6,]

### RG
#### ef_tp
constraint_type = 'ef_tp'
steps = [2]
scales = [2]
val_ckpt_dir = 'data/pretrained/value/val-H192O2-Max-ltls_until_4-epoch\\=0000-val_loss\\=0.73119.ckpt'
val_tag = '73119'


@click.command()
@click.option("--work-dir", type=str, default=BASE_DIR)
@click.option("--gpu-id", type=int, default=0)
@click.option("--guider", '-g', type=str, required=True)
@click.option("--rg-n-ltl-train", type=int, default=30)
@click.option("--product-params", '-p', is_flag=True)
def main(
    work_dir: str, 
    gpu_id: int,
    guider: str,
    rg_n_ltl_train: int,
    product_params: bool,
):
    ## check guider
    assert guider in GUIDERS, f"Unkown guider type {guider}; guider must be one of {GUIDERS}"
    
    ## check param list
    if product_params:
        ### Cartetian product of step and scale spaces
        params = product(steps, scales)
        params_len = len(steps) * len(scales)
    else:
        assert len(steps) == len(scales), "steps and scales must have the same length"
        params = zip(steps, scales)
        params_len = len(steps)
    
    ## check LTL numbers
    if guider == 'rg':
        assert rg_n_ltl_train <= num_ltl, \
            f"rg_n_ltl_train={rg_n_ltl_train} must be less than or equal to "\
            f"num_ltl={num_ltl}"
        rg_n_ltl = {}
        rg_n_ltl['train'] = rg_n_ltl_train
        rg_n_ltl['test'] = num_ltl - rg_n_ltl_train
    
    ## set scripts' file path
    eval_guide_script = os.path.join(work_dir, "eval_H192_pusht_guided_parallel.py")
    eval_baseline_script = os.path.join(work_dir, "eval_H192_pusht_baseline_parallel.py")
    
    # pdb.set_trace()

    ## run
    ### Baseline
    if guider == 'baseline':
        print('\n'+"".join(['*']*70))
        logging.info(f"GPU {gpu_id}: Running \'{guider}\' for {trial} trials...")
        cmd = \
            f"CUDA_VISIBLE_DEVICES={gpu_id} python {eval_baseline_script}  \\\n"\
            f"--config-name eval_H192O2D256_pusht_baseline  \\\n"\
            f"num_ltl={num_ltl}  \\\n"\
            f"trial={trial}  \\\n"\
            f"task.env_runner.max_steps={max_steps}  \\\n"\
            f"task.env_runner.n_envs={n_envs if n_envs is not None else 'null'}  \\\n"\
            f"ltl_dir=\'{ltl_dir}\'  \\\n"\
            f"constraint_type=\'{constraint_type}\'  \\\n"\
            f"output_dir=\'logs/tests/pusht_{guider}_output/H192O2/{constraint_type}/ltl{num_ltl}-{guider}-t{trial}\' \\\n"
        
        logging.info(f"Running command:\n {cmd}")
        os.system(cmd)
    #end if [baseline]
    ### Guided
    elif guider == 'ps' or guider == 'rg':
        #### Compatible with legacy non-standard config_name & output_dir 
        if guider == 'ps':
            config_name = f"eval_H192O2D256_pusht_ps_guide" 
        elif guider == 'rg':
            config_name = f"eval_H192O2D256_pusht_guided"
        #### loop over steps and scales
        for stp, scl in tqdm(params, total=params_len):
            print('\n'+"".join(['*']*70))
            logging.info(f"GPU {gpu_id}: Running \'{guider}\' with stp={stp}, scl={scl}, for {trial} trials...")
            if guider == 'rg':
                for key in ['test', 'train']:
                    n_ltl = rg_n_ltl[key]
                    dir = f'data/pusht/LTLs/ltls_until_4_{key}.txt'
                    cmd = \
                        f"CUDA_VISIBLE_DEVICES={gpu_id} python {eval_guide_script}  \\\n"\
                        f"--config-name {config_name}  \\\n"\
                        f"guider={guider}  \\\n"\
                        f"num_ltl={n_ltl}  \\\n"\
                        f"stp={stp}  \\\n"\
                        f"scl={scl}  \\\n"\
                        f"trial={trial}  \\\n"\
                        f"task.env_runner.max_steps={max_steps} \\\n"\
                        f"task.env_runner.n_envs={n_envs if n_envs is not None else 'null'} \\\n"\
                        f"ltl_dir=\'{dir}\' \\\n"\
                        f"constraint_type=\'{constraint_type}\' \\\n"\
                        f"output_dir=\'logs/tests/pusht_{guider}_output/H192O2/{constraint_type}/{val_tag}/ltl{n_ltl}-{guider}-stp{stp}-scl{scl}-t{trial}\' \\\n"\
                        f"value_checkpoint=\'{val_ckpt_dir}\' \\\n"
                    print(f"Running command:\n {cmd}")
                    os.system(cmd)
                #end for [key]
            elif guider == 'ps':
                cmd = \
                f"CUDA_VISIBLE_DEVICES={gpu_id} python {eval_guide_script}  \\\n"\
                f"--config-name {config_name}  \\\n"\
                f"guider={guider}  \\\n"\
                f"num_ltl={num_ltl}  \\\n"\
                f"stp={stp}  \\\n"\
                f"scl={scl}  \\\n"\
                f"trial={trial}  \\\n"\
                f"task.env_runner.max_steps={max_steps} \\\n"\
                f"task.env_runner.n_envs={n_envs if n_envs is not None else 'null'} \\\n"\
                f"ltl_dir=\'{ltl_dir}\' \\\n"\
                f"constraint_type=\'{constraint_type}\' \\\n"\
                f"output_dir=\'logs/tests/pusht_{guider}_output/H192O2/{constraint_type}/ltl{num_ltl}-{guider}-stp{stp}-scl{scl}-t{trial}\' \\\n"
                print(f"Running command:\n {cmd}")
                os.system(cmd)
            #end if [guider]
        #end for [params]
    else:
        raise ValueError(f"Unkown guider type \'{guider}\'; guider must be one of {GUIDERS}")


if __name__ == '__main__':
    main()

