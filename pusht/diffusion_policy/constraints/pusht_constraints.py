import numpy as np
import torch
import math
from typing import Union

## [., ., block_x(col), block_y(row), angle]
OLD_STATE_MINS = torch.tensor([13.456424, 32.938293, 57.471767, 108.27995, 0.00021559125])
OLD_STATE_MAXS = torch.tensor([496.14618, 510.9579,  439.9153,  485.6641,  6.2830877])
STATE_MINS = torch.tensor([0.65232, 1.23263, 4.13197, 4.57473, 0.000018104653], dtype=torch.float64)
STATE_MAXS = torch.tensor([510.67751, 510.95900, 496.99951, 507.15730, 6.283179], dtype=torch.float64)
STATE_BOUNDS_MIN = torch.tensor([  0,   0,   0,   0, 0], dtype=torch.float64)
STATE_BOUNDS_MAX = torch.tensor([512, 512, 512, 512, 2*math.pi], dtype=torch.float64)
GOAL_POSE = torch.tensor([256.00, 256.00, 0.79], dtype=torch.float64)
EPS = 1e-8

def normalize(s, norm):
    return 2*(s-norm[0])/(norm[1]-norm[0])-1

def get_rectangle_boundry(center, norm, radius=0.5):
    """
        center: (row, col)
        norm:   {'row': (min, max),
                 'col': (min, max),}
    """
    row, col = center[0], center[1]
    if isinstance(radius, float):
        radius = (radius, radius)

    def p_bottom(x):
        return normalize(row+radius[0], norm['row']) - x[:,2]

    def p_up(x):
        return x[:,2] - normalize(row-radius[0], norm['row'])

    def p_left(x):
        return x[:,3] - normalize(col-radius[1], norm['col'])

    def p_right(x):
        return normalize(col+radius[1], norm['col']) - x[:,3]

    return [p_bottom, p_up, p_left, p_right]


### constraints for agent
def agent_in_circles(center=[256,256], rad=64., normalized=False):
    """
    Define and return a list of constraints for the block-in-circle test.
    """
    assert isinstance(rad, float)
    def in_circle(st:torch.tensor, normalized=normalized, 
                  custom_normalizer=None):
        """
        Args:
            st (torch.tensor): (time_steps, feature, batch)
            normalized (bool): whether the input is normalized already; 
                               default is False. 
        """
        dim = slice(0, 2)

        if custom_normalizer is None:
            normed_center = normalize(torch.tensor(center), 
                                    [STATE_BOUNDS_MIN[dim], STATE_BOUNDS_MAX[dim]] )
            ### radius only needs scaling
            normed_rad = 2*torch.tensor([rad, rad]) / (STATE_BOUNDS_MAX[dim] - STATE_BOUNDS_MIN[dim])
        else:
            normed_center = custom_normalizer.normalize(torch.tensor(center))
            normed_rad = custom_normalizer.normalize(torch.tensor([rad, rad]))
        if not normalized: 
            ## normalization
            normed_x = normalize(st[:, 0], 
                                [STATE_BOUNDS_MIN[0], STATE_BOUNDS_MAX[0]] )
            normed_y = normalize(st[:, 1], 
                                [STATE_BOUNDS_MIN[1], STATE_BOUNDS_MAX[1]] )
        #end if
        else: 
            ## no need for built-in normalization
            normed_x = st[:, 0]
            normed_y = st[:, 1]

        dx_sqr = torch.square(normed_x - normed_center[0]) / torch.square(normed_rad[0])
        dy_sqr = torch.square(normed_y - normed_center[1]) / torch.square(normed_rad[1])

        d = torch.tensor(1.) - torch.sqrt(dx_sqr + dy_sqr)
        assert torch.le(d, torch.tensor(1.)).min()>0, "d should be less than 1!"
        return d
    
    return [in_circle]


ef_params = [
    [(260, 255), 20.], 
    [(305, 200), 25.], 
    [(280, 330), 30.], 
    [(160, 320), 25.], 
]

ef_con_groups = [
    agent_in_circles(center=np.array(p[0]), rad=p[1]) for p in ef_params
]


constraint_param_dict = {
    'ef_oa': ef_params,
    'ef_tp': ef_params,
}

constraint_dict = {
    'ef_oa': ef_con_groups,
    'ef_tp': ef_con_groups,
}

### check completeness
for key in constraint_param_dict.keys():
    assert constraint_dict.get(key) is not None, \
        f"Param key \'{key}\' not found in constraint_dict"
#end for
# for key in constraint_dict.keys():
#     assert constraint_param_dict.get(key) is not None, \
#         f"Constraint key \'{key}\' not found in constraint_param_dict"
# #end for
