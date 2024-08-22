norm_umaze_v1   = {'row': (0.39, 3.22),
                   'col': (0.63, 3.22),}
norm_medium_v1  = {'row': (0.67, 6.22),
                   'col': (0.48, 6.22),}
norm_large_v1   = {'row': (0.4, 7.22),
                   'col': (0.44, 10.22),}

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

centers_umaze_v1 = [(0.55,3.05),
                    (1.05,3.05),
                    (1.55,3.05),
                    (2.05,3.05),
                    (2.55,3.05),
                    (3.05,3.05),]
con_groups_umaze_v1 = [get_rectangle_boundry(center, norm_umaze_v1, radius=0.25) for center in centers_umaze_v1]

centers_medium_v1 = [(1.95,1.15),
                     (2.65,3.15),
                     (1.95,4.45),
                     (3.45,1.95),
                     (4.15,3.65),
                     (3.65,4.45),]
con_groups_medium_v1 = [get_rectangle_boundry(center, norm_medium_v1, radius=0.25) for center in centers_medium_v1]

centers_large_v1 = [(3,2),
                    (2,4),
                    (2,6),
                    (6,6),
                    (5,7),
                    (5,9),]
con_groups_large_v1 = [get_rectangle_boundry(center, norm_large_v1, radius=0.5) for center in centers_large_v1]


con_groups = {'maze2d-umaze-v1': con_groups_umaze_v1,
              'maze2d-medium-v1': con_groups_medium_v1,
              'maze2d-large-v1': con_groups_large_v1,}
