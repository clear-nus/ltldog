from typing import Dict, Sequence, Union, Optional
from gym import spaces
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
from diffusion_policy.constraints.pusht_constraints import constraint_param_dict
import numpy as np
import cv2

class PushTKeypointsLTLEnv(PushTKeypointsEnv):
    def __init__(self,
            constraint_key,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog, 
            damping=damping,
            render_size=render_size,
            keypoint_visible_rate=keypoint_visible_rate, 
            agent_keypoints=agent_keypoints,
            draw_keypoints=draw_keypoints,
            reset_to_state=reset_to_state,
            render_action=render_action,
            local_keypoint_map= local_keypoint_map, 
            color_map= color_map)
        
        self.state_trj = None
        self.constraint_key = constraint_key
        self.circle_traps = constraint_param_dict[self.constraint_key]
        self.max_spawn_trials = 1000
        self.ltl_str = ""


    def reset(self):
        obs = super().reset()        
        ## Keep resetting if spawned in traps
        t = 0
        while t < self.max_spawn_trials:
            flg = self._tee_in_traps() or self._agent_in_traps()
            if not flg:
                break
            else:
                obs = super().reset()
                t += 1
                # print(f"block pos: {self.block.position}")
        # end while
        if flg:
            print(f"UserWarning: Init positions in trap!")
        
        st = self._get_state()
        self.state_trj = np.atleast_2d(st)

        return obs


    def step(self, action):
        _ = super().step(action)
        
        st = self._get_state()
        self.state_trj = np.concatenate([self.state_trj, np.atleast_2d(st)], axis=0)
        
        return _

    def set_ltl_str(self, ltl_str:str):
        self.ltl_str = ltl_str

    def get_state(self):
        return self._get_state()

    def get_state_trj(self):
        if self.state_trj is None:
            return None
        return self.state_trj.copy()

    def _agent_in_traps(self) -> bool:
        is_in = False
        pos_agent = np.array(self.agent.position)
        for trap in self.circle_traps:
            c = np.array(trap[0])
            r = trap[1]
            d2 = np.sum(np.square(pos_agent - c))
            if d2 < r**2:
                is_in = True
                break
        #end checking all traps
        
        return is_in

    def _tee_in_traps(self) -> bool:
        is_in = False
        pos_tee = np.array(self.block.position)
        for trap in self.circle_traps:
            c = np.array(trap[0])
            r = trap[1]
            d2 = np.sum(np.square(pos_tee - c))
            if d2 < r**2:
                is_in = True
                break
        #end checking all traps
        
        return is_in


    def _get_state(self):
        ## reconstruct the full state trajectory
        st = np.array(self.agent.position)
        st = np.concatenate([
                st, 
                np.array(list(self.block.position) + [self.block.angle])
            ], 
            axis=-1
        )
        st = st.reshape((1, 5))
        return st


    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))

        # st = self._get_state()
        
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step, 
            # 'state': st,
        }

        return info
    
    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        
        img = self.draw_tee_pos(img)
        img = self.draw_action_trj(img)
        img = self.draw_traps(img)
        img = self.draw_ltl_str(img)

        return img


    def draw_traps(self, img):
        rsz = self.render_size
        thickness = int(1.5/ rsz *self.render_size)
        clr = (255,0,0)
        for i, trap in enumerate(self.circle_traps): 
            # print(trap)
            c = (np.array(trap[0]) / 512 * rsz).astype(np.int32)
            r = int((trap[1]) / 512 * rsz)
            cv2.circle(
                img, 
                c, 
                int(r),
                color=clr,
                thickness=thickness
            )
            cv2.putText(
                img = img,
                text = str(i),
                org = c,
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.5,
                color = clr,
                thickness = int(thickness * 1.5)
            )
        
        return img


    def draw_tee_pos(self, img):
        rsz = self.render_size
        pos_tee = np.array(self.block.position)
        coord = (pos_tee / 512 * rsz).astype(np.int32)
        marker_size = int(8/rsz*self.render_size)
        thickness = int(1.2/rsz*self.render_size)
        cv2.drawMarker(img, coord,
            color=(255,0,255), markerType=cv2.MARKER_CROSS,
            markerSize=marker_size, thickness=thickness)
        return img


    def draw_action_trj(self, img):
        rsz = self.render_size
        if self.state_trj is None:
            return img

        thickness = int(1.0/ rsz *self.render_size)
        rad = int(5 / 512 *  rsz)
        
        actions = self.state_trj[:, :2].copy()
        actions = (actions / 512 *  rsz).astype(np.int32)
        n_points = len(actions)

        # Create a colormap and get colors for each point
        colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        colors = colormap[np.linspace(0, 255, n_points, endpoint=False, dtype=np.uint8)]
        
        # Draw each point on the image
        for color, xy in zip(colors, actions):
            cv2.circle(
                img, 
                xy, 
                rad, 
                tuple(int(c) for c in color[0]), 
                thickness=thickness
            )
        #end for
        
        return img


    def draw_ltl_str(self, img):
        # print(f"LTL: \'{self.ltl_str}\'")
        rsz = self.render_size
        thickness = int(1.5/ rsz *self.render_size)
        clr = (0,0,0)
        cv2.putText(
                img = img,
                text = self.ltl_str,
                org = (np.array([30, 480]) / 512 * rsz).astype(np.int32),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.4,
                color = clr,
                thickness = int(thickness * 1.5)
            )

        return img
