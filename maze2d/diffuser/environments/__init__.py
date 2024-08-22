from .registration import register_environments

registered_environments = register_environments()


from gym.envs.registration import register

from .maze_model import LARGE_EMPTY, LARGE_HASH

register(
    id='maze2d-large-empty-v1',
    entry_point='diffuser.environments.maze_model:ResetEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':LARGE_EMPTY,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 6.7,
        'ref_max_score': 273.99
    }
)

register(
    id='maze2d-large-hash-v1',
    entry_point='diffuser.environments.maze_model:ResetEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':LARGE_HASH,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 6.7,
        'ref_max_score': 273.99
    }
)
