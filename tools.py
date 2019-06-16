import matplotlib.pyplot as plt

def paint_wall_by_points_set(ax = None, points_set = None):
    if ax is None:
        for i in range(len(points_set)):
            st = i
            ed = (i + 1) % len(points_set)
            x_set = [points_set[st][0], points_set[ed][0]]
            y_set = [points_set[st][1], points_set[ed][1]]
            plt.plot(x_set, y_set)
    else:
        for i in range(len(points_set)):
            st = i
            ed = (i + 1) % len(points_set)
            x_set = [points_set[st][0], points_set[ed][0]]
            y_set = [points_set[st][1], points_set[ed][1]]
            z_set = [points_set[st][2], points_set[ed][2]]
            ax.plot3D(x_set, y_set, z_set)
    return

def paint_wall_by_2_corners_3d(ax, left_down_corner, right_up_corner):
    x_left = left_down_corner[0]
    x_right = right_up_corner[0]
    y_down = left_down_corner[1]
    y_up = right_up_corner[1]
    z_low = left_down_corner[2]
    z_high = right_up_corner[2]
    point_set = []
    # low
    point_set.append([x_left, y_down, z_low])
    point_set.append([x_right, y_down, z_low])
    point_set.append([x_right, y_up, z_low])
    point_set.append([x_left, y_up, z_low])
    point_set.append([x_left, y_down, z_low])
    # high
    point_set.append([x_left, y_down, z_high])
    point_set.append([x_right, y_down, z_high])
    point_set.append([x_right, y_up, z_high])
    point_set.append([x_left, y_up, z_high])
    point_set.append([x_left, y_down, z_high])
    # left
    point_set.append([x_left, y_down, z_low])
    point_set.append([x_left, y_up, z_low])
    point_set.append([x_left, y_up, z_high])
    point_set.append([x_left, y_down, z_high])
    point_set.append([x_left, y_down, z_low])
    # right
    point_set.append([x_right, y_down, z_low])
    point_set.append([x_right, y_up, z_low])
    point_set.append([x_right, y_up, z_high])
    point_set.append([x_right, y_down, z_high])
    point_set.append([x_right, y_down, z_low])
    # front
    point_set.append([x_right, y_up, z_low])
    point_set.append([x_left, y_up, z_low])
    point_set.append([x_left, y_up, z_high])
    point_set.append([x_right, y_up, z_high])
    point_set.append([x_right, y_up, z_low])
    # back
    point_set.append([x_right, y_down, z_low])
    point_set.append([x_left, y_down, z_low])
    point_set.append([x_left, y_down, z_high])
    point_set.append([x_right, y_down, z_high])
    point_set.append([x_right, y_down, z_low])

    paint_wall_by_points_set(ax = ax, points_set = point_set)
    return

def paint_wall_by_2_corners_2d(left_down_corner, right_up_corner):
    x_left = left_down_corner[0]
    x_right = right_up_corner[0]
    y_down = left_down_corner[1]
    y_up = right_up_corner[1]
    point_set = []
    point_set.append([x_left, y_down])
    point_set.append([x_right, y_down])
    point_set.append([x_right, y_up])
    point_set.append([x_left, y_up])
    paint_wall_by_points_set(ax = None, points_set = point_set)
