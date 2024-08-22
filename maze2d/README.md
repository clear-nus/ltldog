# `Maze2d`: Diffuser-based Implementation 


## Installation

Install `ltldog` as a package in a newly created environment

```sh
pip install -e .
```


## Datasets and environments
We use the official offline datasets and environments of Maze2d from D4RL.

### $\textnormal{LTL}_f$ objective
The full set of $\textnormal{LTL}_f$ constraints are listed at [`datasets/ltls_until.txt`](./datasets/ltls_until.txt)

### Configuration
The default configuration of regions of interest for atomic propositions are listed at [`config/maze2d_constraints.py`](./config/maze2d_constraints.py)


## Training
The default path for logs and model saving is at `logs`.

### Diffusion model
Train a diffusion model with:
```sh
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
```
#### Configs
The default hyperparameters are listed in [`config/maze2d.py`](config/maze2d.py).
You can override any of them with runtime flags, eg `--batch_size 64`.

### Value model
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