import matplotlib.pyplot as plt
import time
from simulator import ParticleWaterSimulatorEasy
import tools

particles_num = 20
timestep = 0.01
gravity = 9.8
space_left_down_corner = (0.0, 0.0)
space_right_up_corner = (10.0, 10.0)
collision_test = True

# simulator init
sim = ParticleWaterSimulatorEasy(
    particle_nums_ = particles_num,
    timestep_= timestep,
    space_left_down_corner_= space_left_down_corner,
    space_right_up_corner_ = space_right_up_corner,
    gravity_= gravity,
    collision_detect_ = collision_test)

plt.ion()
saveid = 0
while True:
    t1 = time.time()
    # do timestep
    points = sim.dotimestep()

    # paint points
    plt.scatter(points[0], points[1], s = 5)


    # paint wall
    wall_x_dist = space_right_up_corner[0] - space_left_down_corner[0]
    wall_y_dist = space_right_up_corner[1] - space_left_down_corner[1]
    plt.xlim(space_left_down_corner[0] - wall_x_dist/2, space_right_up_corner[0] + wall_x_dist/2)
    plt.ylim(space_left_down_corner[1] - wall_y_dist/2, space_right_up_corner[1] + wall_y_dist/2)
    tools.paint_wall_by_2_corners(space_left_down_corner, space_right_up_corner)

    # title
    plt.title("FPS: %d, particle_num: %d, cur_time = %.2f" % (int(1.0/(time.time() - t1 + 1e-6)), particles_num, sim.get_cur_time()))
    # print('[log][display] simulator cost %.3f s, display cost %.3f s' % ((t2 - t1), (t_last - t2)))
    saveid += 1

    # pause
    plt.pause(1e-5)
    plt.cla()

