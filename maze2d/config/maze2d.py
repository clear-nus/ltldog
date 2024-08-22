import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'dim_mults': (1, 4, 8),
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'loss_type': 'l2',
        'clip_denoised': True,

        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),
        'device': 'cuda',

        ## training
        'n_steps_per_epoch': 10000,
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,

        ## planning
        'batch_size_plan': 1,
        'num_avg': 5,
        'vis_freq': 10,
    },

    'values': {
        ## model
        'model': 'models.ValueClassifier',
        'dim_mults': (1, 4, 8),
        'diffusion': 'models.ValueDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'loss_type': 'value_l2',
        'clip_denoised': True,

        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.LTLsValueDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'use_padding': False,
        'max_path_length': 40000,
        ## value-specific kwargs
        'termination_penalty': None,
        'normed': False,

        ## serialization
        'logbase': 'logs',
        'prefix': 'value/',
        'exp_name': watch(diffusion_args_to_watch),
        'device': 'cuda',

        ## training
        'n_steps_per_epoch': 10000,
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'bucket': None,
    },

    'test': {
        'batch_size': 1,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'logbase': 'logs',
        'prefix': 'tests/release',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 10,
        # 'suffix': '0',
        'device': 'cuda',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'rg_value_loadpath': 'f:value/H{horizon}_T{n_diffusion_steps}',

        'diffusion_epoch': 'latest',
        'rg_value_epoch': 'latest',

        ## guide sample_kwargs
        'n_guide_steps': {'default':2},
        'scale': {'default':0.1},
        't_stopgrad': {'default':2},
        'scale_grad_by_std': {'default':True},
    },
}

#------------------------ overrides ------------------------#

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 256,
        'n_diffusion_steps': 256,

        'n_steps_per_epoch': 500,
        'n_train_steps': 125_000,
        'batch_size': 512,

        'sample_freq': 200,
        'n_samples': 9,
    },
    'values': {
        'horizon': 256,
        'n_diffusion_steps': 256,

        'n_steps_per_epoch': 200,
        'n_train_steps': 125_000,
        'batch_size': 512,
    },
    'test': {
        'horizon': 256,
        'n_diffusion_steps': 256,

        'n_guide_steps': {'rg':10, 'dps':10},
        'scale': {'rg':50, 'dps':10},
        't_stopgrad': {'rg':2, 'dps':2},
        'scale_grad_by_std': {'rg':False, 'dps':False},
    },
}

maze2d_medium_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,

        'n_steps_per_epoch': 500,
        'n_train_steps': 125_000,
        'batch_size': 512,

        'sample_freq': 200,
        'n_samples': 9,
    },
    'values': {
        'horizon': 384,
        'n_diffusion_steps': 256,

        'n_steps_per_epoch': 200,
        'n_train_steps': 125_000,
        'batch_size': 512,
    },
    'test': {
        'horizon': 384,
        'n_diffusion_steps': 256,

        'n_guide_steps': {'rg':10, 'dps':10},
        'scale': {'rg':50, 'dps':10},
        't_stopgrad': {'rg':2, 'dps':2},
        'scale_grad_by_std': {'rg':False, 'dps':False},
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 512,
        'n_diffusion_steps': 256,

        'n_steps_per_epoch': 500,
        'n_train_steps': 125_000,
        'batch_size': 512,

        'sample_freq': 200,
        'n_samples': 9,
    },
    'values': {
        'horizon': 512,
        'n_diffusion_steps': 256,

        'n_steps_per_epoch': 200,
        'n_train_steps': 125_000,
        'batch_size': 512,
    },
    'test': {
        'horizon': 512,
        'n_diffusion_steps': 256,

        'n_guide_steps': {'rg':10, 'dps':10},
        'scale': {'rg':50, 'dps':10},
        't_stopgrad': {'rg':2, 'dps':2},
        'scale_grad_by_std': {'rg':False, 'dps':False},
    },
}
