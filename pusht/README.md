# `PushT`: Diffusion-Policy-based Implementation

**Table of Contents**
- [PushT: Diffusion-Policy-based Implementation](#ltldog-diffusion-policy-based-implementation)
  - [**Installation**](#installation)
  - [Play with the Env (Optional)](#play-with-the-env-optional)
    - [PushT Task Demo](#pusht-task-demo)
  - [Training](#training)
    - [Dataset](#dataset)
      - [PushT Trajectory Dataset](#pusht-trajectory-dataset)
      - [LTL formulas](#ltl-formulas)
      - [Value Dataset](#value-dataset)
    - [Configs](#configs)
      - [Training Configs](#training-configs)
      - [LTL Atomic Propositions](#ltl-atomic-propositions)
    - [Strat Training](#strat-training)
  - [Inference](#inference)
    - [Inference Configs](#inference-configs)


## **Installation**

1. Create a Conda environment with `conda_environment.yaml` (we also recommend Mambaforge) 

    ```sh
    mamba env create -f conda_environment.yaml
    # Or you can use conda
    # mamba env create -f conda_environment.yaml
    ```

    **Remarks**: The `dgl` package installed using the YAML file is only for CUDA version of 11.8. *If you are using a different CUDA version*, please try:

    ```sh
    # Try the following if you are using CUDA other than 11.8
    ## Uninstall the cu118 ver. DGL
    mamba activate ltldogdp
    pip uninstall dgl
    ## Install the right one
    ### replace "cu1xx" with the CUDA version you use, e.g., cu116, cu117, cu121, etc.
    pip install dgl==1.1.1 -f https://data.dgl.ai/wheels/cu1xx/repo.html
    ```

    If it doesn't work, please proceed to [DGL official site](https://www.dgl.ai/pages/start.html) for more support. 

2. Install `diffusion_policy` as a package in your newly created environment

    ```sh
    mamba activate ltldogdp 
    pip install -e . 
    ```


## Play with the Env (Optional)

### PushT Task Demo

The demo script inherits from Diffusion Policy. 
Familiarize yourself with the environment by 

```sh
python demo_pusht.py --help
```


## Training 

### Dataset

- Default data are stored at `data/` (you can use symbolic links if needed): 

    ```sh
    mkdir -p ./data/pusht/LTLs  ./data/pretrained/diffusion ./data/pretrained/value 
    cp ltl_txt/*.txt data/pusht/LTLs/
    ```

    Now the LTL formulas used in the paper are copied. 

- Training logs and saved checkpoints are at `data/outputs/`. 

#### PushT Trajectory Dataset

- Trajectory datasets should be put under `data/pusht` for training. 
- The original PushT dataset used in Diffusion Policy should be available [here](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip).  
- For the *augmented* trajectory dataset used in our paper, download from [Google Drive](https://drive.google.com/file/d/1jRfJtiBL-cYcFbmQczzVGXK8THSp--en/view?usp=sharing). 

#### LTL formulas

See `.txt` files under `data/pusht/LTLs/`. See more [below](#ltl-atomic-propositions). 


#### Value Dataset 

For <span style="font-variant: small-caps;">LTLDoG-R</span> variant, we need to train a regressor model with LTL satisfaction values. 
These values are calculated given the *trajectory dataset* and the *LTL<sub>f</sub> formulas*. 

- Values can be calculated by our script. 
**Remember to check the source codes before executing.**

    ```sh
    python scripts/generate_pusht_value_dataset.py
    ```


### Configs

The configuration setting pipeline inherits from <span style="font-variant: small-caps;">DiffusionPolicy</span>. 

#### Training Configs

Training configs are located at `diffusion_policy/config/`. Adjust dataset path in corresponding subconfigs under `diffusion_policy/config/tasks/`. 

#### LTL Atomic Propositions


In our experiments, each atomic proposition (AP, denoted as "pX" in LTL formulas) represents a region in the ambient state space. 
These regions are defined in `diffusion_policy/constraints/pusht_constraints.py`. 

Users may devise and configure customized regions, not limitted to circles, by defining proper parameterization of the regions and implementing *differentiable* value functions that can determine the truth of an AP (positive value for True and negative for False by default). 
Check the script for an intuition. 


### Strat Training

Simply call the training script with a desired config file. *E.g.*:

- For training the vanilla Diffusion Policy (as the diffusion backbone of <span style="font-variant: small-caps;">LTLDoG</span>):

    ```sh
    python scripts/train.py --config-name=H16O2A8D100_train_diffusion_unet_lowdim_workspace
    ```

- For training a value regressor: 

    ```sh
    python scripts/train.py --config-name=H16O2A8_train_pusht_ef_no_value
    ```

**Reminder**:

- Using the pretrained model released from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) is not enough; the model should be trained over an augmented dataset with more abundunt behaviors. See the appendix of our paper for explanation. 
- Move the trained checkpoints from `data/outputs` to `data/pretrained/diffusion` or `data/pretrained/value` for inference later. 



## Inference

Evaluation scripts are under `scripts/`:

- Run `eval_H\d_pusht_*.py` with a proper config file for inference. 
- Calling examples: 

    ```sh
    # LTLDoG-S
    python scripts/eval_H16_pusht_guided_parallel.py  --config-name eval_H16_pusht_ps_guide 
    ```

    ```sh
    # LTLDoG-R
    ## with a trained regressor model configured in the config file
    python scripts/eval_H16_pusht_guided_parallel.py  --config-name eval_H16_pusht_guided 
    ```

- There are two versions of models: `H16` and `H192` (for obstacle avoidance and temporal tasks, respectively). Check the scripts' source codes for details. 

There are also sequential executing scripts that could be used to run multiple times (for different parameters), see `eval_H\d_seq.py` for details. 

- Calling example: 

    ```sh
    python scripts/eval_H16_seq.py  --gpu-id=0 --guider=rg
    ```

    Argument `--guider` should be one of {rg, ps, baseline}. Explore more settings in the script. 

Results are recorded by default under `logs/tests/`. 

### Inference Configs

- Configuration files `diffusion_policy/config/eval_*.yaml`. 
- Configure the configs to adjust parameters and **make sure calling the right config file** when launching eval scripts. 
- Parameters in the config file will be executed while parameters set in the eval scripts are *only for naming purpose*. Remember to **adjust both** to avoid unecesssary confusion.
