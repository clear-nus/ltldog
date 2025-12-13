# `Maze2d`: Diffuser-based Implementation 


## Installation

Install dependencies (e.g. python 3.8):

```sh
pip install -r requirements.txt
```
<!--
Follow the instructions from mujoco_py and then install
```sh
pip install mujoco-py==2.0.2.5
```
-->
**Remarks**: The `dgl` package installed here is only for CUDA version of 11.8. *If you are using a different CUDA version, please refer to the official website of DGL for more support.*

Install `ltldog` as a package in the newly created environment:

```sh
pip install -e .
```
<!--
For new versions of setuptools, might need the following command for compatibility:
```sh
pip install -e . --config-settings editable_mode=compat
```
-->

## Datasets and environments
We use the official offline datasets and environments of Maze2d from D4RL.

### $\textnormal{LTL}_f$ objective
The full set of $\textnormal{LTL}_f$ constraints are listed at [`datasets/ltls_until.txt`](./datasets/ltls_until.txt)

### Configuration
The default configuration of regions of interest for atomic propositions are listed at [`config/maze2d_constraints.py`](./config/maze2d_constraints.py)


## Training
The default path for logs and model saving is at `logs`.

### 1. Diffusion model
Train a diffusion model with:
```sh
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
```
#### Configs
The default hyperparameters are listed in [`config/maze2d.py`](config/maze2d.py).
You can override any of them with runtime flags, e.g. `--batch_size 64`.

### 2. Value model
The LTLDoG-R variant of our method requires training a formula value regression neural network on noised versions of input trajectory:
```sh
python scripts/train_values.py --config config.maze2d --dataset maze2d-large-v1
```
This training process will save a numpy file at `datasets` that stores all the formula values of all trajectories. Check the _get_values() method in file [sequence](./diffuser/datasets/sequence.py) for details. This file will be loaded when trained next time.


## Testing

Test using the diffusion model with:
```sh
python scripts/test_maze2d.py --config config.maze2d --dataset maze2d-large-v1
```
Results will be stored in a `json` file. You can override `diffusion_loadpath` and `rg_value_loadpath` in [`config/maze2d.py`](config/maze2d.py).