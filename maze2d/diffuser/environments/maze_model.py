from d4rl.pointmaze.maze_model import MazeEnv

LARGE_EMPTY = \
        "############\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "############"

LARGE_HASH = \
        "############\\"+\
        "#O#O#O#O#O##\\"+\
        "#OOOOOOOOOO#\\"+\
        "#O#O#O#O#O##\\"+\
        "#OOOOOOOOOO#\\"+\
        "#O#O#O#O#O##\\"+\
        "#OOOOOOOOOO#\\"+\
        "#O#O#O#O#O##\\"+\
        "############"

class ResetEnv(MazeEnv):

    def set_obs(self, obs):
        qpos = obs[:self.model.nq]
        qvel = obs[self.model.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()
