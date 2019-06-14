import matplotlib.pyplot as plt
import time
from simulator import ParticleWaterSimulatorEasy, ParticleWaterSimulatorSPH
import tools
import os
import shutil

# render config
save_dir = "./imgs/"
# simulator config
particles_num = 500
timestep = 0.01
gravity = 9.8
space_left_down_corner = (0.0, 0.0)
space_right_up_corner = (100.0, 100.0)
collision_test = True

# simulator init
# sim = ParticleWaterSimulatorEasy(
#     particle_nums_ = particles_num,
#     timestep_= timestep,
#     space_left_down_corner_= space_left_down_corner,
#     space_right_up_corner_ = space_right_up_corner,
#     gravity_= gravity,
#     collision_detect_ = collision_test)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)


# d = max(((25.0 / 500) / 3.14)**0.5, 1)
d = 1
sim = ParticleWaterSimulatorSPH(
    particle_nums_ = particles_num,
    timestep_= timestep,
    space_left_down_corner_= space_left_down_corner,
    space_right_up_corner_ = space_right_up_corner,
    gravity_= gravity,
    collision_detect_ = collision_test,
    kernel_poly6_d_= d)

# plt.ion()
saveid = 0
t1 = time.time()
while True:
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
    plt.title("FPS: %.1f, particle_num: %d, cur_time = %.2f" % ((1.0/(time.time() - t1 + 1e-6)), particles_num, sim.get_cur_time()))
    t1 = time.time()
    # print('[log][display] simulator cost %.3f s, display cost %.3f s' % ((t2 - t1), (t_last - t2)))
    saveid += 1

    # pause
    # plt.pause(1e-5)
    plt.savefig(save_dir + str(saveid))
    plt.cla()

