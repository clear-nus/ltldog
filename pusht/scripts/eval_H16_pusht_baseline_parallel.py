"""
Usage:
python eval_H16_pusht_baseline_parallel.py --config-name eval_H16_pusht_baseline

--diffusion_checkpoint  data/pretrained/diffusion/dp-H16-O2-A8-epoch=0700-test_mean_score=0.911.ckpt

-o logs/tests/pusht_rg_output/
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import pdb
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.ast_builder import ASTBuilder
from diffusion_policy.common.ltl_parser import LTLParser, LTLParseError
from diffusion_policy.dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator
from diffusion_policy.constraints.pusht_constraints import constraint_dict, constraint_param_dict
from omegaconf import OmegaConf
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


DEVICE = 'cuda'
# ltl_dir = "data/pusht/LTLs/ltls_no_4.txt"



def get_policy(cfg):
    ## Resolve config to get main workspace
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    ## get policy from workspace
    policy = workspace.model

    ## load diffusion model checkpoint
    diffusion_payload = torch.load(open(cfg.diffusion_checkpoint, 'rb'), pickle_module=dill)
    diffusion_cfg = diffusion_payload['cfg']
    diffusion_cls = hydra.utils.get_class(diffusion_cfg._target_)
    diffusion_workspace = diffusion_cls(diffusion_cfg, output_dir=cfg.output_dir)
    diffusion_workspace: BaseWorkspace
    diffusion_workspace.load_payload(diffusion_payload, exclude_keys=None, include_keys=None)
    ## get diffusion model from workspace
    diffusion_policy = diffusion_workspace.model
    if diffusion_cfg.training.use_ema:
        diffusion_policy = diffusion_workspace.ema_model
    #end if
    diffusion_model = diffusion_policy.model
    
    ## Replacing diffusion model with the pretrained model
    policy.model = diffusion_model
    
    ## Set new policy's normalizer as the one in diffusion policy.
    policy.set_normalizer(diffusion_workspace.model.normalizer)

    device = torch.device(DEVICE)
    policy.to(device)
    policy.eval()

    return policy


#------------------------------------ main -----------------------------------#
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'diffusion_policy','config'))
)

def main(cfg):
    num_ltl = cfg.num_ltl
    atomic_cons = constraint_dict[cfg.constraint_type]
    output_dir = cfg.output_dir
    ltl_dir = cfg.ltl_dir

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #--------------------------- policy ----------------------------#
    policy = get_policy(cfg)
    
    #--------------------------- tester ----------------------------#
    props = ['p0','p1','p2','p3']
    with open(ltl_dir) as file:
        ltls=file.read()
        ltls=ltls.split("\n")[0:-1]
    ltls = ltls
    ltls = ltls[:num_ltl]
    n_ltl = len(ltls)
    n_train_ltls = int(0.8*n_ltl)
    # pdb.set_trace()
    tester_dtl = DTL_Cont_Cons_Evaluator(device='cuda')
    tester_dtl.set_props_cons_and_ltl(atomic_cons, props, ltls[0])
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(
        policy=policy, 
        ltl_formulas=ltls,  
        tester_dtl=tester_dtl, 
        ltl_graphs=None
    )

    try:
        runner_log['constraint_type'] = cfg.constraint_type
        runner_log['constraint_params'] = constraint_param_dict[cfg.constraint_type]
    except RuntimeError as err:
        print(f"[WARNING] Logging error:", err)
    
    # dump log to json
    json_log = dict()
    raw_data_to_save = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif isinstance(value, np.ndarray):
            raw_data_to_save[key] = value
            json_log[key] = os.path.join(output_dir, f"raw_data.npz/{key}")
        else:
            try:
                json_log[key] = value
            except RuntimeError as err:
                print(f"[WARNING] Logging error:", err)
    out_path = os.path.join(output_dir, 'eval_baseline_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    ## save data to npz
    np.savez_compressed(os.path.join(output_dir, 'raw_data.npz'), **raw_data_to_save)


if __name__ == '__main__':
    main()
