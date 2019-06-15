import matplotlib.pyplot as plt
import time
from simulator import ParticleWaterSimulatorEasy, ParticleWaterSimulatorSPH
import tools
import os
import shutil

if __name__ == '__main__':
    # render config
    img_save_dir = "./imgs/"
    record_save_dir = "./record/"

    # simulator config
    particles_num = 2300
    particles_num = int(particles_num ** 0.5) ** 2
    timestep = 0.003
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

    # img 每次清空
    if os.path.exists(img_save_dir):
        shutil.rmtree(img_save_dir)
    os.mkdir(img_save_dir)

    # record 每次不清空
    if  False == os.path.exists(record_save_dir):
        os.mkdir(record_save_dir)

    # 这里需要和下面保持一致
    space = (space_right_up_corner[0] - space_left_down_corner[0]) * (space_right_up_corner[1] - space_left_down_corner[1]) / 4
    d = max((( space / particles_num) / 3.14)**0.5, 1)

    sim = ParticleWaterSimulatorSPH(
        particle_nums_ = particles_num,
        timestep_= timestep,
        space_left_down_corner_= space_left_down_corner,
        space_right_up_corner_ = space_right_up_corner,
        gravity_= gravity,
        collision_detect_ = collision_test,
        kernel_poly6_d_= d,
        record_save_dir_ = record_save_dir)

    plt.ion()
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
        plt.title("FPS: %.1f, particle_num: %d, cur_time = %.2f" % ((1.0/(time.time() - t1 + 1e-6)), sim.get_particle_num(), sim.get_cur_time()))
        t1 = time.time()
        # print('[log][display] simulator cost %.3f s, display cost %.3f s' % ((t2 - t1), (t_last - t2)))

        # pause
        plt.pause(1e-5)
        # plt.savefig(img_save_dir + str(sim.get_frameid()))
        plt.cla()
