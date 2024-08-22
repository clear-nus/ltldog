import os
import sys
import json
import logging
import click
import pdb
import numpy as np
from tqdm import tqdm
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# # use line-buffering for both stdout and stderr
# sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
# sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

BASE_DIR = str(os.path.dirname(os.path.abspath(__file__)))
GUIDERS = ['baseline', 'ps', 'rg']


## Parameters
num_ltl = 10
trial = 50
ltl_dir = 'data/pusht/LTLs/ltls_no_4.txt'
constraint_type = 'ef_oa'
max_steps = 304
n_envs = None
steps = [1,2,5,7]
scales = [s/10. for s in range(2, 10, 2)]
skips = [(2,1), (5,1)]


@click.command()
@click.option("--work-dir", type=str, default=BASE_DIR)
@click.option("--gpu-id", type=int, default=0)
@click.option("--guider", '-g', type=str, required=True)
@click.option("--product-params", '-p', is_flag=True)
def main(
    work_dir: str, 
    gpu_id: int,
    guider: str,
    product_params: bool,
    ):
    ## check guider
    assert guider in GUIDERS, f"guider must be one of {GUIDERS}"
    ## set scripts' file path
    eval_guide_script = os.path.join(work_dir, "eval_H16_pusht_guided_parallel.py")
    eval_baseline_script = os.path.join(work_dir, "eval_H16_pusht_baseline_parallel.py")
    
    ## check param list
    if product_params:
        ### Cartetian product of step and scale spaces
        params = product(steps, scales)
        params_len = len(steps) * len(scales)
    else:
        assert len(steps) == len(scales), "steps and scales must have the same length"
        params = zip(steps, scales)
        params_len = len(steps)

    ## run
    ### Baseline
    if guider == 'baseline':
        print("".join(['*']*70))
        logger.info(f"GPU {gpu_id}: Running \'{guider}\' for {trial} trials...")
        cmd = \
            f"CUDA_VISIBLE_DEVICES={gpu_id} python {eval_baseline_script}  \\\n"\
            f"--config-name eval_H16_pusht_baseline  \\\n"\
            f"num_ltl={num_ltl}  \\\n"\
            f"trial={trial}  \\\n"\
            f"task.env_runner.max_steps={max_steps}  \\\n"\
            f"task.env_runner.n_envs={n_envs if n_envs is not None else 'null'}  \\\n"\
            f"ltl_dir=\'{ltl_dir}\'  \\\n"\
            f"constraint_type=\'{constraint_type}\'  \\\n"\
            f"output_dir=\'logs/tests/pusht_{guider}_output/H16O2A8/{constraint_type}/ltl{num_ltl}-{guider}-t{trial}\' \\\n"
        logger.info(f"Running command:\n {cmd}")
        os.system(cmd)
    #end if [baseline]
    ### Guided
    elif guider == 'ps' or guider == 'rg':
        #### Compatible with legacy non-standard config_name & output_dir 
        if guider == 'ps':
            config_name = f"eval_H16_pusht_ps_guide" 
        elif guider == 'rg':
            config_name = f"eval_H16_pusht_guided"
        #### loop over steps and scales
        for stp, scl in tqdm(params, total=params_len):
            param_np = np.array([stp, scl])
            for skip in skips:
                if np.linalg.norm(param_np - np.array(skip)) < 1e-6:
                    print(f"Skipping step={stp}, scale={scl}")
                    continue
            #end for [skip]
            print("".join(['*']*70))
            logger.info(f"GPU {gpu_id}: Running \'{guider}\' with stp={stp}, scl={scl}, for {trial} trials...")
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
                f"output_dir=\'logs/tests/pusht_{guider}_output/H16O2A8/{constraint_type}/ltl{num_ltl}-{guider}-stp{stp}-scl{scl}-t{trial}\' \\\n"
            print(f"Running command:\n {cmd}")
            os.system(cmd)
    else:
        raise ValueError(f"Unkown guider type \'{guider}\'; guider must be one of {GUIDERS}")


if __name__ == '__main__':
    main()
