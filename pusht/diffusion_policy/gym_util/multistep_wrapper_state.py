import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper, aggregate, dict_take_last_n


class MultiStepWrapperPushTFullState(MultiStepWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_trj = np.empty((0, 5))
        self.base_env = self.unwrapped

    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()
        st_trj = self.base_env.get_state_trj()
        assert st_trj is not None, \
            f"Initial state of the env is not properly set."
        self.state_trj = st_trj

        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = \
                super(MultiStepWrapper, self).step(act)

            st_trj = self.base_env.get_state_trj()
            assert st_trj is not None, \
                f"Initial state of the env is not properly set."
            self.state_trj = st_trj

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info


    def run_dill_function(self, dill_fn, **kwargs):
        fn = dill.loads(dill_fn)
        return fn(self, **kwargs)
    
