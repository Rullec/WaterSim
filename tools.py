import matplotlib.pyplot as plt

def paint_wall_by_points_set(points_set):
    for i in range(len(points_set)):
        st = i
        ed = (i + 1) % len(points_set)
        x_set = [points_set[st][0], points_set[ed][0]]
        y_set = [points_set[st][1], points_set[ed][1]]
        plt.plot(x_set, y_set)
    return

def paint_wall_by_2_corners(left_down_corner, right_up_corner):
    if len(left_down_corner) == 2:
        x_left = left_down_corner[0]
        x_right = right_up_corner[0]
        y_down = left_down_corner[1]
        y_up = right_up_corner[1]
        point_set = []
        point_set.append([x_left, y_down])
        point_set.append([x_right, y_down])
        point_set.append([x_right, y_up])
        point_set.append([x_left, y_up])
        paint_wall_by_points_set(point_set)
    elif len(left_down_corner) == 3:
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
        #back
        point_set.append([x_right, y_down, z_low])
        point_set.append([x_left, y_down, z_low])
        point_set.append([x_left, y_down, z_high])
        point_set.append([x_right, y_down, z_high])
        point_set.append([x_right, y_down, z_low])

        paint_wall_by_points_set(point_set)