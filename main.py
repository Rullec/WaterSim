import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from simulator import ParticleWaterSimulatorEasy, ParticleWaterSimulatorSPH, ParticleWaterSimulatorBase
import tools
import os
import shutil

if __name__ == '__main__':
    # display mode or record mode
    record = True
    display = False

    # render config
    img_save_dir = "./imgs/"
    record_save_dir = "./record/"

    # simulator config
    simulator_dimension = 3
    particles_num = 3000

    # particles_num = int(particles_num ** 0.5) ** 2
    timestep = 0.002
    gravity = 9.8
    space_left_down_corner = None
    space_right_up_corner = None
    cube_edge = 30.0

    if simulator_dimension == 2:
        space_left_down_corner = (0.0, 0.0)
        space_right_up_corner = (cube_edge, cube_edge)
    elif simulator_dimension == 3:
        space_left_down_corner = (0.0, 0.0, 0.0)
        space_right_up_corner = (cube_edge, cube_edge, cube_edge)
    else:
        assert 0 == 1

    collision_test = True
    multi_processor = False

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

    # remove stastistic
    statistic_path ="./log/stastistic.txt"
    if os.path.exists(statistic_path):
        os.remove(statistic_path)
    with open(statistic_path, "w") as f:
        f.write("********************\n")
    # 这里需要和下面保持一致
    space = 0
    if simulator_dimension == 2:
        space = (space_right_up_corner[0] - space_left_down_corner[0]) * (space_right_up_corner[1] - space_left_down_corner[1]) / 4
    elif simulator_dimension == 3:
        space = (space_right_up_corner[0] - space_left_down_corner[0]) * (space_right_up_corner[1] - space_left_down_corner[1]) * (space_right_up_corner[2] - space_left_down_corner[2])
    d = 2.5
    # d = max((( space / particles_num) / 3.14)**0.5, 1)
    # d = (( space / particles_num) / 3.14)**0.5
    print(d)
    # sim = ParticleWaterSimulatorBase(
    #     simulator_base_dimension_ = simulator_dimension,
    #     particle_nums_ = particles_num,
    #     timestep_= timestep,
    #     space_left_down_corner_= space_left_down_corner,
    #     space_right_up_corner_ = space_right_up_corner,
    #     gravity_= gravity,
    #     collision_detect_ = collision_test,
    #     # kernel_poly6_d_= d,
    #     record_save_dir_ = record_save_dir,
    #     multi_processor_ = multi_processor)

    sim = ParticleWaterSimulatorSPH(
        simulator_base_dimension_ = simulator_dimension,
        particle_nums_ = particles_num,
        timestep_= timestep,
        space_left_down_corner_= space_left_down_corner,
        space_right_up_corner_ = space_right_up_corner,
        gravity_= gravity,
        collision_detect_ = collision_test,
        kernel_poly6_d_= d,
        record_save_dir_ = record_save_dir,
        multi_processor_ = multi_processor)

    if display == True:
        plt.ion()
    t1 = time.time()
    if simulator_dimension == 3:
        fig = plt.figure()
        ax = Axes3D(fig)  # fig.add_subplot(111, projection = '3d')
        while True:
            points = sim.dotimestep()

            wall_x_dist = space_right_up_corner[0] - space_left_down_corner[0]
            wall_y_dist = space_right_up_corner[1] - space_left_down_corner[1]
            wall_z_dist = space_right_up_corner[2] - space_left_down_corner[2]

            # paint the wall
            tools.paint_wall_by_2_corners_3d(ax, space_left_down_corner, space_right_up_corner)

            ax.scatter(points[0], points[1], points[2])
            ax.set_xlim(space_left_down_corner[0] - wall_x_dist / 4, space_right_up_corner[0] + wall_x_dist / 4)
            ax.set_ylim(space_left_down_corner[1] - wall_y_dist / 4, space_right_up_corner[1] + wall_y_dist / 4)
            ax.set_zlim(space_left_down_corner[2] - wall_y_dist / 4, space_right_up_corner[2] + wall_z_dist / 4)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # ax.set_title("123")
            ax.set_title("FPS: %.2f, particle_num: %d, cur_time = %.3f"% ((1.0 / (time.time() - t1 + 1e-6)), sim.get_particle_num(), sim.get_cur_time()))

            # ax.set_title("FPS: ")# % ((1.0 / (time.time() - t1 + 1e-6)), sim.get_particle_num(), sim.get_cur_time()))
            t1 = time.time()
            if display == True:
                plt.pause(1e-5)
            if record == True:
                plt.savefig(img_save_dir + str(sim.get_frameid()))

            ax.cla()
    elif simulator_dimension == 2:

        while True:
            points = sim.dotimestep()
            plt.scatter(points[0], points[1], s = 5)

            # paint wall
            wall_x_dist = space_right_up_corner[0] - space_left_down_corner[0]
            wall_y_dist = space_right_up_corner[1] - space_left_down_corner[1]
            plt.xlim(space_left_down_corner[0] - wall_x_dist / 2, space_right_up_corner[0] + wall_x_dist / 2)
            plt.ylim(space_left_down_corner[1] - wall_y_dist / 2, space_right_up_corner[1] + wall_y_dist / 2)

            tools.paint_wall_by_2_corners_2d(space_left_down_corner, space_right_up_corner)

            # title
            plt.title("FPS: %.2f, particle_num: %d, cur_time = %.3f" % ((1.0/(time.time() - t1 + 1e-6)), sim.get_particle_num(), sim.get_cur_time()))
            t1 = time.time()
            # print('[log][display] simulator cost %.3f s, display cost %.3f s' % ((t2 - t1), (t_last - t2)))

            # pause
            if display == True:
                plt.pause(1e-5)
            if record == True:
                plt.savefig(img_save_dir + str(sim.get_frameid()))
            plt.cla()
