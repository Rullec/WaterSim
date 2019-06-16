import time
import logging
import numpy as np
import multiprocessing
from multiprocessing import Pool
import sys

'''
    Class ParticleWaterSimulator
        this class is used to simulator a particle system

    simulation formulas(Explicit Euler):
    q means:        point position (2*1)
    q_vel means:    point velocity (2*1)
    q_acc means:    point acceleration  (2*1)
    
    "DOTIMESTEP" IS THE CORE FUNCTION FOR SIMULATION PROCEDURE.
'''
class Emmiter:
    emit_pos = None
    emit_mode = None
    emit_point_mass = 1
    emit_base_dimentsion = -1

    def __init__(self, emit_pos_):
        assert emit_pos_.shape == (2, ) or emit_pos_.shape == (3, )
        self.emit_base_dimentsion = emit_pos_.shape[0]
        self.emit_pos = emit_pos_
        self.emit_mode = "linear"

    def blow(self, point_num):
        point_pos = np.reshape(self.emit_pos, (self.emit_base_dimentsion, 1)).repeat(point_num, axis = 1)
        point_vel = None
        point_acc = None
        point_mass = None
        if "linear" == self.emit_mode:
            single_vel = np.ones(self.emit_base_dimentsion) * 10
            point_vel = np.reshape(single_vel, (self.emit_base_dimentsion, 1)).repeat(point_num, axis = 1)
            point_acc = np.zeros([self.emit_base_dimentsion, point_num])
            point_mass = np.ones(point_num) * self.emit_point_mass

        return point_pos, point_vel, point_acc, point_mass

class ParticleWaterSimulatorBase:
    # 2D simulator or 3D simulator?
    simulator_base_dimension = 2

    # simulation params
    cur_time = 0.0
    timestep = 0.0
    frameid = 0
    particles_num = -1
    space_left_down_corner = None
    space_right_up_corner = None
    g = 0
    collision_detect = False

    # simulation space
    space_length = 0
    space_height = 0
    space_width = 0

    # collision penalty force coeff
    collision_epsilon = -1
    collision_penalty_k = 5e3  # control the distance
    collision_penalty_b = 1  # control the velocity

    # damping coeff
    damping_coeff = 1

    # status varibles
    point_pos = np.zeros([simulator_base_dimension, 0])
    point_vel = np.zeros([simulator_base_dimension, 0])
    point_acc = np.zeros([simulator_base_dimension, 0])
    point_mass = np.zeros(0)

    # system cost time record
    time_cost_dotimestep = 0.0
    time_cost_compute_force = 0.0
    time_cost_collision_test = 0.0

    # emiiter
    emmiter1 = None
    emit_amount = None

    # logging module
    logger = None
    log = None

    # record mode
    record = False
    record_dir = None
    record_filename = None

    # SP or MP, it will decide by the simulator automatically
    MULTIPLEPROCESS = -1
    multipleprocess_num = -1
    multipleprocess_infolist = []

    def __init__(self,
                 simulator_base_dimension_,
                 particle_nums_,
                 timestep_,
                 space_left_down_corner_,
                 space_right_up_corner_,
                 gravity_,
                 collision_detect_,
                 record_save_dir_,
                 multi_processor_):
        '''

        :param particle_nums_:
        :param timestep_:
        :param space_left_down_corner_:
        :param space_right_up_corner_:
        :param gravity_:
        '''
        # logging module init
        log_filename = "./log/" + str(time.strftime("%Y-%m-%d %H%M%S", time.localtime())) + str('.txt')
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s  - %(message)s")
        fh = logging.FileHandler(filename=log_filename)  # ouput log both the the console and the file
        logger = logging.getLogger(self.__class__.__name__)
        logger.addHandler(fh)
        logger.info('[SimulatorBase] Init begin')

        # system property
        # x_num = int(particle_nums_ ** 0.5)
        # y_num = int(particle_nums_ ** 0.5)
        # self.particles_num = x_num * y_num
        self.simulator_base_dimension = simulator_base_dimension_
        self.particles_num = particle_nums_
        self.point_pos = np.zeros([self.simulator_base_dimension, self.particles_num])
        self.point_vel = np.zeros([self.simulator_base_dimension, self.particles_num])
        self.point_acc = np.zeros([self.simulator_base_dimension, self.particles_num])
        self.point_mass = np.ones(self.particles_num)
        self.space_left_down_corner = space_left_down_corner_
        self.space_right_up_corner = space_right_up_corner_
        assert  len(self.space_left_down_corner) == self.simulator_base_dimension
        assert  len(self.space_right_up_corner) == self.simulator_base_dimension

        # self.point_pos[0, :] = space_length / 2 * np.tile(np.linspace(0, 1, num = x_num), y_num) + space_left_down_corner_[
        #     0] + space_length / 4
        # self.point_pos[1, :] = space_height / 2 * np.repeat(np.linspace(0, 1, num = y_num), x_num) + space_left_down_corner_[
        #     1] + space_height / 3
        self.init_points_variables(space_left_down_corner_, space_right_up_corner_)


        # simulation property
        self.g = gravity_
        self.timestep = timestep_
        self.collision_detect = collision_detect_
        if self.simulator_base_dimension == 2:
            self.collision_epsilon = min(self.space_height, self.space_length) / 100
        elif self.simulator_base_dimension == 3:
            self.collision_epsilon = min(self.space_height, self.space_length, self.space_width) / 100

        # record mode init
        self.record_dir = record_save_dir_
        self.record_filename = self.record_dir + str(time.strftime("%Y-%m-%d %H%M%S", time.localtime())) + str('.txt')

        # init emitter
        if self.simulator_base_dimension ==  2:
            self.emmiter1 = Emmiter(np.array([0, 0]))
        elif self.simulator_base_dimension == 3:
            self.emmiter1 = Emmiter(np.array([0, 0, 0]))
        self.emit_amount = 1

        # MP computation setting
        # if self.particles_num > 1300:
        self.MULTIPLEPROCESS = multi_processor_
        self.multipleprocess_num = max(int((self.particles_num / 5000) * multiprocessing.cpu_count()), 4)

        # print('particle_num = %d' % self.particles_num)
        # print('timestep = %d' % self.timestep)
        # print('active space = (%f, %f) - (%f, %f)' % (
        #     space_left_down_corner_[0], space_left_down_corner_[1], self.space_right_up_corner[0],
        #     self.space_right_up_corner[1]))

        # print('******************Simulator Init Succ****************')
        logger.info('[SimulatorBase] Init succ')
        return

    def init_points_variables(self, space_right_up_corner_, space_left_down_corner_):
        # init these points
        if self.simulator_base_dimension == 2:
            self.space_length = space_right_up_corner_[0] - space_left_down_corner_[0]
            self.space_height = space_right_up_corner_[1] - space_left_down_corner_[1]
            self.point_pos[0, :] = self.space_length / 2 * np.random.rand( self.particles_num) + space_left_down_corner_[
                0] + self.space_length / 4
            self.point_pos[1, :] = self.space_height / 2 * np.random.rand( self.particles_num) + space_left_down_corner_[
                1] + self.space_height / 4
        elif self.simulator_base_dimension == 3:
            self.space_length = space_right_up_corner_[0] - space_left_down_corner_[0]
            self.space_height = space_right_up_corner_[1] - space_left_down_corner_[1]
            self.space_width = space_right_up_corner_[2] - space_left_down_corner_[2]
            self.point_pos[0, :] = self.space_length / 2 * np.random.rand( self.particles_num) + space_left_down_corner_[
                0] + self.space_length / 4
            self.point_pos[1, :] = self.space_height / 2 * np.random.rand( self.particles_num) + space_left_down_corner_[
                1] + self.space_height / 4
            self.point_pos[2, :] = self.space_width / 2 * np.random.rand(self.particles_num) + space_left_down_corner_[
                1] + self.space_width / 10
        else:
            raise("the dimension is illegal")

    def dotimestep(self):
        # dynamic simulation - compute total forces and acceleration
        st = time.time()

        # time++
        self.cur_time += self.timestep
        self.frameid += 1
        self.time_cost_dotimestep = time.time() - st
        return self.point_pos

    # get and set methods
    def get_cur_time(self):
        return self.cur_time

    def get_frameid(self):
        return self.frameid

    def get_particle_num(self):
        return self.particles_num

    # update simulation state
    # currently, the explicit euler method
    def update_state(self):
        '''
            q_vel = q_vel + timestep * q_accel
            q_pos = q_pos + timestep * q_accel
        '''
        self.point_vel += self.timestep * self.point_acc
        self.point_pos += self.timestep * self.point_vel

    def do_collision_test_between_wall_and_particles_3d(self):
        collision_force = np.zeros([3, self.particles_num])
        return collision_force

    def do_collision_test_between_wall_and_particles_2d(self):
        st = time.time()
        collision_force = np.zeros([2, self.particles_num])
        x_left = self.space_left_down_corner[0]
        x_right = self.space_right_up_corner[0]
        y_down = self.space_left_down_corner[1]
        y_up = self.space_right_up_corner[1]

        for i in range(self.particles_num):
            # for a point i
            pos_i = self.point_pos[:, i]
            vel_i = self.point_vel[:, i]
            next_pos = pos_i + self.timestep * vel_i

            if abs(pos_i[0] - x_left) < self.collision_epsilon:
                collision_force[:, i][0] += abs(pos_i[0] - x_left) * self.collision_penalty_k
                if vel_i[0] < 0:
                    collision_force[:, i][0] += np.abs(vel_i[0]) * self.collision_penalty_b
                # print('[debug][collision] %d point collision with x' % i)

            if abs(pos_i[0] - x_right) < self.collision_epsilon:
                collision_force[:, i][0] += -(abs(pos_i[0] - x_right) * self.collision_penalty_k)
                if vel_i[0] > 0:
                    collision_force[:, i][0] += -abs(vel_i[0]) * self.collision_penalty_b
                # print('[debug][collision] %d point collision with x' % i)

            if abs(pos_i[1] - y_down) < self.collision_epsilon:
                collision_force[:, i][1] += abs(pos_i[1] - y_down) * self.collision_penalty_k
                if vel_i[1] < 0:
                    collision_force[:, i][1] += np.abs(vel_i[1]) * self.collision_penalty_b

            if abs(pos_i[1] - y_up) < self.collision_epsilon:
                collision_force[:, i][1] += -abs(pos_i[1] - y_up) * self.collision_penalty_k
                if vel_i[1] > 0:
                    collision_force[:, i][1] += -np.abs(vel_i[1]) * self.collision_penalty_b
        return collision_force

    def do_collision_test_between_wall_and_particles(self):
        '''
            @Function: do_collision_test_between_wall_and_particles
                this function is used to:
                1. detect the collision (judgement according to some creatia collision for or not)
                2. compute the collision penalty force accordly

                Now this function can only handle a box boundary, limited to the big computation amout
            @params: None
            @return: collision force
            @date: 12/06/2019
        '''

        # print('[log][simulation] do collision test')
        collision_force = None
        if self.simulator_base_dimension == 2:
            collision_force = self.do_collision_test_between_wall_and_particles_2d()
        elif self.simulator_base_dimension == 3:
            print('[warning] the collision force in 3d hasn\'t been implemented')
            collision_force = self.do_collision_test_between_wall_and_particles_3d()

        return collision_force

    def compute_damping_force(self):
        return -1 * self.point_vel * self.damping_coeff

    def compute_gravity_force(self):
        assert self.g > 0

        points_gravity = None
        if self.simulator_base_dimension == 2:
            points_gravity = np.zeros([2, self.particles_num])
            points_gravity[1,] = -1 * self.g * self.point_mass
        elif self.simulator_base_dimension == 3:
            points_gravity = np.zeros([3, self.particles_num])
            points_gravity[2,] = -1 * self.g * self.point_mass
        else:
            raise ("the dimension is illegal")
        return points_gravity

    def update_multipleprocessor_infolist(self):
        '''
            compute the info list for multiple processor division
        :return:
        '''
        # judge whether to use MP or not
        # if self.particles_num > 1300:
        #     self.MULTIPLEPROCESS = True
        #     self.multipleprocess_num = multiprocessing.cpu_count()

        # update the particle num division
        self.multipleprocess_infolist = []
        for i in range(self.multipleprocess_num):
            if 0 == i:
                st_id = 0
                ed_id = int((i + 1) / self.multipleprocess_num * self.particles_num)
            elif i == self.multipleprocess_num - 1:
                st_id = ed_id
                ed_id = self.particles_num
            else:
                st_id = ed_id
                ed_id = int((i + 1) / self.multipleprocess_num * self.particles_num)
            if st_id > ed_id:
                st_id = ed_id
            para = np.array([i, st_id, ed_id])
            self.multipleprocess_infolist.append(para)

    def record_data(self):
        with open(self.record_filename, 'a') as f:
            f.write("%d %d " % (self.frameid, self.particles_num))
            for i in range(self.particles_num):
                if self.simulator_base_dimension == 2:
                    f.write("%.5f %.5f " % (self.point_pos[0, i], self.point_pos[1, i]))
                elif self.simulator_base_dimension == 3:
                    f.write("%.5f %.5f %.5f " % (self.point_pos[0, i], self.point_pos[1, i], self.point_pos[2, i]))
            f.write("\n")

    def emitter_inject(self):
        self.particles_num += self.emit_amount
        pos, vel, acc, mass = self.emmiter1.blow(self.emit_amount)

        self.point_pos = np.concatenate((self.point_pos, pos), axis=1)
        self.point_vel = np.concatenate((self.point_vel, vel), axis=1)
        self.point_acc = np.concatenate((self.point_acc, acc), axis=1)
        self.point_mass = np.append(self.point_mass, mass)

        # you must update the division info after add or diminish some vertex
        self.update_multipleprocessor_infolist()

class ParticleWaterSimulatorEasy(ParticleWaterSimulatorBase):

    # lennard jones forces coef
    lennard_jones_k1 = 0.01
    lennard_jones_k2 = 0.01
    lennard_jones_m = 4
    lennard_jones_n = 2

    def __init__(self,simulator_base_dimension_, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_,
                 collision_detect_, record_save_dir_, multi_processor_):
        ParticleWaterSimulatorBase.__init__(self,simulator_base_dimension_, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_,
                 collision_detect_, record_save_dir_, multi_processor_)
        return

    def dotimestep(self):
        # dynamic simulation - compute total forces and acceleration
        st = time.time()
        points_force = self.compute_forces()

        # compute accel
        # self.point_acc[0,] = points_force[0,] * (1.0 / self.point_mass)
        # self.point_acc[1,] = points_force[1,] * (1.0 / self.point_mass)
        self.point_acc = np.multiply(points_force, 1.0 / self.point_mass)
        assert self.point_acc.shape == (self.simulator_base_dimension, self.particles_num)

        # update the state - forward euler
        self.update_state()

        # time++
        self.cur_time += self.timestep
        self.frameid += 1
        self.time_cost_dotimestep = time.time() - st
        print('[log][simulator] do timestep, cur time = %.3f s, cur frameid = %d' % (self.cur_time, self.frameid))
        print('[log][simulator] dotimestep cost %.5f s, jones force cost %.5f s' % (
        self.time_cost_dotimestep, self.time_cost_jones_force))
        # print(self.point_vel.dtype)
        return self.point_pos


    # compute the lennard jones forces for particle system
    '''
        Lennard_Jones force is aimed at pushing 2 particles far away from each other
        f(xi, xj) = ( k1 / (|xi - xj|^m) - k2 / (|xi - xj|)^n)
                    *
                    (xi - xj) / |xi - xj| 
    '''
    def compute_lennard_jones_force(self):
        st = time.time()
        jones_force = np.zeros([self.simulator_base_dimension, self.particles_num])
        for pi in range(self.particles_num):
            for pj in range(pi + 1, self.particles_num):
                pos_xi = self.point_pos[:, pi]
                pos_xj = self.point_pos[:, pj]
                xi_xj_dist = np.linalg.norm(pos_xi - pos_xj, ord = 2)
                force_coeff = (
                        self.lennard_jones_k1 / xi_xj_dist ** self.lennard_jones_m - self.lennard_jones_k2 / xi_xj_dist ** self.lennard_jones_n)
                force_xi_xj = force_coeff / xi_xj_dist * (pos_xi - pos_xj)
                jones_force[:, pi] += force_xi_xj
                jones_force[:, pj] += -force_xi_xj
        ed = time.time()
        self.time_cost_jones_force = ed - st

        jones_force = np.clip(jones_force, a_max = 100, a_min = -100)
        # print('[log][jones_force] cost time %.3f s' % (ed-st))
        return jones_force

    # Function: self.compute_forces
    def compute_forces(self):
        '''
            this function is aimed at computing total forces for the whole particle system
        :return:
        '''
        st = time.time()
        points_force = np.zeros([self.simulator_base_dimension, self.particles_num])

        ## gravity computation
        points_gravity = self.compute_gravity_force()

        # jone forces computation
        points_jone_forces = self.compute_lennard_jones_force()

        # damping
        points_damping_forces = self.compute_damping_force()

        # collision detect
        if self.collision_detect == True:
            collision_force = self.do_collision_test_between_wall_and_particles()
        else:
            collision_force = np.zeros([self.simulator_base_dimension, self.particles_num])

        # total force summary
        points_force += points_gravity
        points_force += points_jone_forces
        points_force += points_damping_forces
        points_force += collision_force

        # print(collision_force[1,])
        self.time_cost_compute_force = time.time() - st

        # print(points_damping_forces)
        return points_force

class ParticleWaterSimulatorSPH(ParticleWaterSimulatorBase):

    # kernel parameters
    kernel_poly6_d = -1
    kernel_poly6_coeff = -1

    # sph variables
    sph_point_density = -1
    sph_point_pressure = -1

    # constant
    gas_constant = 8.314

    # simulation variables
    viscosity_coeff = 1e-3 # the viscosity of water is 1e-3

    def __init__(self, simulator_base_dimension_, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_,
                 collision_detect_, kernel_poly6_d_, record_save_dir_,multi_processor_):
        ParticleWaterSimulatorBase.__init__(self, simulator_base_dimension_, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_,
                 collision_detect_, record_save_dir_,multi_processor_)

        # init sph variable
        self.kernel_poly6_d = kernel_poly6_d_
        self.kernel_poly6_coeff = 315.0 / (64.0 * np.pi * (self.kernel_poly6_d ** 9))

        # print('[log][simulator] ParticleWaterSimulatorSPH init succ')
        return

    def dotimestep(self):
        st = time.time()

        # update Multiple Processor usage info
        self.update_multipleprocessor_infolist()

        # emit
        self.emitter_inject()

        # compute force
        points_force = self.compute_force()

        # compute accel
        self.point_acc = np.multiply(points_force, 1.0 / self.point_mass)
        # self.point_acc[0,] = points_force[0,] * (1.0 / self.point_mass)
        # self.point_acc[1,] = points_force[1,] * (1.0 / self.point_mass)

        # update the state - forward euler
        self.update_state()

        # time++
        self.cur_time += self.timestep
        self.frameid += 1
        self.time_cost_dotimestep = time.time() - st
        print('[log][simulator] do timestep, cur time = %.3f s, cur frameid = %d' % (self.cur_time, self.frameid))

        # record
        self.record_data()
        return self.point_pos

    def compute_force(self):
        st = time.time()
        sum_force = np.zeros([self.simulator_base_dimension, self.particles_num])

        # 1.1 compute the point density
        st1 = time.time()
        self.compute_point_density()
        st2 = time.time()
        time_compute_density = st2 - st1

        # 1.2 compute the pressure (not force) in eache point
        st1 = time.time()
        self.compute_point_pressure()
        st2 = time.time()
        time_compute_point_pressure = st2 - st1

        # 1.3 compute the pressure force for each point
        st1 = time.time()
        pressure_force = self.compute_pressure_force()
        st2 = time.time()
        time_compute_pressure = st2 - st1

        # 1.4 compute the viscosity force
        st1 = time.time()
        viscosity_force = self.compute_viscosity_force()
        st2 = time.time()
        time_viscosity = st2 - st1

        # 1.5 compute the gravity
        gravity = self.compute_gravity_force()

        # 1.6 compute the collision force
        # print('collsion = ' + str(self.collision_detect))
        st1 = time.time()
        if self.collision_detect == True:
            collision_force = self.do_collision_test_between_wall_and_particles()
        else:
            collision_force = np.zeros([self.simulator_base_dimension, self.particles_num])
        st2 = time.time()
        time_collision = st2 - st1

        # 1.7 compute the damping force
        damping_force = self.compute_damping_force()

        # 1.6 summary
        sum_force += pressure_force
        sum_force += viscosity_force
        sum_force += gravity
        sum_force += collision_force
        sum_force += damping_force

        ed = time.time()
        time_total = ed - st + 1e-6
        print("compute force cost time (%.3f) s, collision %.3f s(%.3f%%), pressure %.3f s(%.3f%%), viscosity %.3f s(%.3f%%), point_density %.3f s(%.3f%%) , point pressure %.3fs(%.3f%%)."
                         % (time_total, time_collision, time_collision/time_total * 100,
                            time_compute_pressure, time_compute_pressure/time_total * 100,
                            time_viscosity, time_viscosity/time_total * 100,
                            time_compute_density, time_compute_density/time_total * 100, time_compute_point_pressure, time_compute_point_pressure/time_total * 100))
        return sum_force

    def compute_sub_viscosity_force(self, para):
        assert para.shape == (3,) # procnum, st_point_id, ed_point_id
        procnum = para[0]
        st_point_id = para[1]
        ed_point_id = para[2]
        # print('st ed = %d %d' % (st_point_id, ed_point_id))
        cur_particles_num = ed_point_id - st_point_id
        viscosity_force = np.zeros([self.simulator_base_dimension, cur_particles_num])

        # compute
        # print("[sub pressure force] density = " + str(self.sph_point_density[-1]))
        for i in range(cur_particles_num):
            id = i + st_point_id
            velocity_i = np.reshape(self.point_vel[:, id], (self.simulator_base_dimension, 1))
            velocity_diff = self.point_vel - velocity_i  # 2*n
            mass_div_density = self.point_mass / self.sph_point_density
            assert mass_div_density.shape == (self.particles_num,)

            velocity_diff_coef_vec = velocity_diff * mass_div_density  # 2 * n
            velocity_diff_coef_vec = velocity_diff_coef_vec.flatten(order='F')
            # velocity_diff_coef_vec = 2n * 1, flatten按列展开
            # (2i， 2i+1)数据对就是第i个点的x y速度差 * 对应系数

            assert velocity_diff_coef_vec.shape == (self.particles_num * self.simulator_base_dimension,)

            # compute the ∇^2_matrix
            pos_i = np.reshape(self.point_pos[:, id], (self.simulator_base_dimension, 1))
            pos_diff = pos_i - self.point_pos

            # the only reason for divide these code into 2 parts is for good names"ddW_dxy2" and "ddW_dxyz2"
            if self.simulator_base_dimension == 2:
                ddW_dxy2 = self.W_poly6_2_order_jacob(pos_diff)
                assert ddW_dxy2.shape == (2, 2 * self.particles_num)

                # compute the result
                viscosity_force[:, i] = self.viscosity_coeff * np.dot(ddW_dxy2, velocity_diff_coef_vec)
            elif self.simulator_base_dimension == 3:
                ddW_dxyz2 = self.W_poly6_2_order_jacob(pos_diff)
                assert ddW_dxyz2.shape == (3, 3 * self.particles_num)

                # compute the result
                viscosity_force[:, i] = self.viscosity_coeff * np.dot(ddW_dxyz2, velocity_diff_coef_vec)
            else:
                assert 0 == 1


        return (procnum, viscosity_force)


    def compute_viscosity_force(self):
        '''
            this function will compute the viscosity force for each point
            and the govern formula is:
                f_viscosity = [f_viscosity_0, ..., f_viscosity_n]_{2*n}
                f_viscosity_i_{2*1} = μ∇^2 v

            f_vis_i = μΣ_j mj * (vj - vi)/ρj *∇^2 W(|xi - xj|)
                    = μΣ_j mj/ρj  * (vj - vi) * ∇^2 W(|xi - xj|)
                    = μΣ_j ∇^2 W(|xi - xj|)_{2*2} * velocity_diff_coef_{2*1}
                    = μ ∇^2_matrix_{2*2n} * velocity_diff_coef_vec{2n*1}
                    = (2*1)


            :return:
        '''
        viscosity_force = np.zeros([self.simulator_base_dimension, self.particles_num])

        if self.MULTIPLEPROCESS == True:
            pool = Pool(self.multipleprocess_num)
            data = pool.map(self.compute_sub_viscosity_force, self.multipleprocess_infolist)
            for i in range(len(data)):
                procnum, force = data[i]
                st_id = self.multipleprocess_infolist[procnum][1]
                ed_id = self.multipleprocess_infolist[procnum][2]
                assert force.shape == (self.simulator_base_dimension, ed_id - st_id)
                viscosity_force[:, st_id: ed_id] = force
            pool.close()
            pool.join()
        else:
            for i in range(self.particles_num):
                # compute the velocity_diff_coef_vec
                velocity_i = np.reshape(self.point_vel[:, i], (self.simulator_base_dimension, 1))
                velocity_diff = self.point_vel - velocity_i  # 2*n or 3*n, it depens
                mass_div_density = self.point_mass / self.sph_point_density
                assert mass_div_density.shape == (self.particles_num, )

                velocity_diff_coef_vec = velocity_diff * mass_div_density   # 2*n or 3*n
                velocity_diff_coef_vec = velocity_diff_coef_vec.flatten(order='F')
                    # velocity_diff_coef_vec = 2n * 1 or 3n * 1, flatten按列展开
                    # 2d case: (2i， 2i+1)数据对就是第i个点的x y速度差 * 对应系数
                    # 3d case: (3i， 3i+1)数据对就是第i个点的x y z速度差 * 对应系数

                assert velocity_diff_coef_vec.shape == (self.particles_num * self.simulator_base_dimension, )

                # compute the ∇^2_matrix
                pos_i = np.reshape(self.point_pos[:, i], (self.simulator_base_dimension, 1))
                pos_diff = pos_i - self.point_pos


                # these following codes are divided into 2 parts is just for simplicit and good name.
                # "ddW_dxy2" and "ddW_dxyz2", they are different
                if self.simulator_base_dimension == 2:
                    ddW_dxy2 = self.W_poly6_2_order_jacob(pos_diff)

                    assert ddW_dxy2.shape == (2, 2 * self.particles_num)

                    # compute the result
                    viscosity_force[:, i] = self.viscosity_coeff * np.dot(ddW_dxy2 , velocity_diff_coef_vec)
                elif self.simulator_base_dimension == 3:
                    ddW_dxyz2 = self.W_poly6_2_order_jacob(pos_diff)

                    assert ddW_dxyz2.shape == (3, 3 * self.particles_num)

                    # compute the result
                    viscosity_force[:, i] = self.viscosity_coeff * np.dot(ddW_dxyz2, velocity_diff_coef_vec)
                # np.set_printoptions(linewidth=200, floatmode='fixed')
                # print('ddW_dxy = ' + str(ddW_dxy2))
                # print('coef = ' + str(velocity_diff_coef_vec))
        # print('viscosity force = ' + str(viscosity_force))
        return viscosity_force

    def compute_sub_pressure_force(self, para):
        assert para.shape == (3,) # procnum, st_point_id, ed_point_id
        procnum = para[0]
        st_point_id = para[1]
        st_point_id = para[1]
        ed_point_id = para[2]
        # print('st ed = %d %d' % (st_point_id, ed_point_id))
        cur_particles_num = ed_point_id - st_point_id
        pressure_force = np.zeros([self.simulator_base_dimension, cur_particles_num])

        # compute
        # print("[sub pressure force] density = " + str(self.sph_point_density[-1]))
        for i in range(cur_particles_num):
            id = i + st_point_id
            coeff_vector = self.point_mass * \
                           (self.sph_point_pressure + self.sph_point_pressure[id]) / (2 * self.sph_point_density)
            assert coeff_vector.shape == (self.particles_num,)

            # compute the ∇W(|xi - xj|)
            pos_diff = np.reshape(self.point_pos[:, id], (self.simulator_base_dimension, 1)) - self.point_pos

            pressure_force_i = np.zeros(self.simulator_base_dimension)
            if self.simulator_base_dimension == 2:
                dW_dxy = self.W_poly6_1_order_gradient(pos_diff)

                assert dW_dxy.shape == (2, self.particles_num)

                # compute the pressure_i
                pressure_force_i = -np.dot(dW_dxy, coeff_vector)
            elif self.simulator_base_dimension == 3:
                dW_dxyz = self.W_poly6_1_order_gradient(pos_diff)

                assert dW_dxyz.shape == (3, self.particles_num)

                # compute the pressure_i
                pressure_force_i = -np.dot(dW_dxyz, coeff_vector)

            # compute the pressure_i
            pressure_force[:, i] = pressure_force_i
        return (procnum, pressure_force)

    def compute_pressure_force(self):
        '''
            Function: compute_pressure
                this function is aimed at computing pressure for ith point, its formula:
                pressure_i = -  Σj mj * (pi + pj) / 2 * pj * ∇W(|xi - xj|) = (2, 1) or (3, 1)
                           = -  Σj coeff_j * ∇W(|xi - xj|) = (2, 1)
                           = - np.dot(∇W(|xi - xj|)_{2*n}, coeff_vec_j_{n*1}) = (2, 1)  or (3, 1)
        '''
        pressure_force = np.zeros([self.simulator_base_dimension, self.particles_num])
        if self.MULTIPLEPROCESS == True:
            # startTime = time.time()

            pool = Pool(self.multipleprocess_num)
            data = pool.map(self.compute_sub_pressure_force, self.multipleprocess_infolist)
            for i in range(len(data)):
                procnum, force = data[i]
                st_id = self.multipleprocess_infolist[procnum][1]
                ed_id = self.multipleprocess_infolist[procnum][2]
                assert force.shape == (self.simulator_base_dimension, ed_id - st_id)
                pressure_force[:, st_id : ed_id] = force
            pool.close()
            pool.join()
            # endTime = time.time()
            # pressure_force_bak = pressure_force.copy()
            # print("MP time : %.3F" % (endTime - startTime))
        else:
            startTime = time.time()
            for i in range(self.particles_num):
                # compute the coeff vector
                coeff_vector = self.point_mass * (self.sph_point_pressure + self.sph_point_pressure[i]) / (2 * self.sph_point_density)
                assert coeff_vector.shape == (self.particles_num, )

                # compute the ∇W(|xi - xj|)
                pos_diff = np.reshape(self.point_pos[:, i], (self.simulator_base_dimension, 1)) - self.point_pos

                pressure_force_i = np.zeros(self.simulator_base_dimension)
                if self.simulator_base_dimension == 2:
                    dW_dxy = self.W_poly6_1_order_gradient(pos_diff)

                    assert dW_dxy.shape == (2, self.particles_num)
                    # if np.linalg.norm(dW_dxy) > 1:
                        # print(' %d th point info' % i)
                        # print('dW_dxy = ' + str(dW_dxy))
                        # print('coeff_vector = ' + str(coeff_vector))
                    # compute the pressure_i
                    pressure_force_i = -np.dot(dW_dxy, coeff_vector)
                elif self.simulator_base_dimension == 3:
                    dW_dxyz = self.W_poly6_1_order_gradient(pos_diff)

                    assert dW_dxyz.shape == (3, self.particles_num)
                    # compute the pressure_i
                    pressure_force_i = -np.dot(dW_dxyz, coeff_vector)
                else:
                    assert  0 == 1
                pressure_force[:, i] = pressure_force_i

            endTime = time.time()
            # print("SP time : %.3F" % (endTime - startTime))
        # print('***************************')
        # print('pressure force = ' + str(pressure_force))
        # print("pressure force")
        # print("%d th frame pressforce = %.3f"  % (self.frameid, np.linalg.norm(pressure_force_bak - pressure_force)))
        return pressure_force

    def compute_sub_point_density(self, para):
        assert para.shape == (3,) # procnum, st_point_id, ed_point_id
        procnum = para[0]
        st_point_id = para[1]
        ed_point_id = para[2]
        # print(para)
        # print('st ed = %d %d' % (st_point_id, ed_point_id))
        cur_particles_num = ed_point_id - st_point_id
        W_xi_xj = np.zeros([self.particles_num, cur_particles_num])
        for i in range(cur_particles_num):
            id = i + st_point_id
            dist = np.reshape(self.point_pos[:, id], (self.simulator_base_dimension, 1)) - self.point_pos
            assert dist.shape == (self.simulator_base_dimension, self.particles_num)
            W_xi_xj[:, i] = self.W_poly6_0_order_constant(dist)
            # print(id)
            # if id == self.particles_num - 1:
            #     print('false Wxixj %d th col = %s' % (id, str(W_xi_xj[:, i])))

        return (procnum, W_xi_xj)

    def compute_point_density(self):
        '''
            this function will compute the point density  "self.sph_point_density" # (it's useful for the computation of 2 forces)
            from the formula:
                ρ(x) = \sum_j mj * W(|x-xj|)
                ρ(xi)_{1*1} = np.dot(self.point_mass_{1*n}, W(|xi-xj|)_{n*1})
                ρ(x)_{1*n} = np.dot(self.point_mass_{1*n}, W(|xi-xj|)_{n*n})
        :return: None
        '''

        W_xi_xj = np.zeros([self.particles_num, self.particles_num])
        # W_xi_xj_bak = None
        # sph_point_density_bak = None
        if self.MULTIPLEPROCESS == True:
            pool = Pool(self.multipleprocess_num)
            data = pool.map(self.compute_sub_point_density, self.multipleprocess_infolist)
            pool.close()
            pool.join()
            for i in range(len(data)):
                procnum, Wij = data[i]
                # print(procnum)
                st_id = self.multipleprocess_infolist[procnum][1]
                ed_id = self.multipleprocess_infolist[procnum][2]
                assert Wij.shape == (self.particles_num, ed_id - st_id)
                W_xi_xj[:, st_id : ed_id] = Wij
            assert W_xi_xj.shape == (self.particles_num, self.particles_num)
            # W_xi_xj_bak = W_xi_xj.copy()
            # print('false W_xixj = ' + str(W_xi_xj))

            # sph_point_density_bak = np.dot(self.point_mass, W_xi_xj)
            # print(W_xi_xj)
        else:
            # compute W(|xi-xj|_{n*n})
            for i in range(W_xi_xj.shape[1]):
                dist = np.reshape(self.point_pos[:, i], (self.simulator_base_dimension, 1)) - self.point_pos
                assert dist.shape == (self.simulator_base_dimension, self.particles_num)
                W_xi_xj[:, i] = self.W_poly6_0_order_constant(dist)
            # print(W_xi_xj)

            # compute the point density
            assert self.point_mass.shape == (self.particles_num, )
            # print(self.point_mass.shape)
            # print('true W_xixj = ' + str(W_xi_xj))
        self.sph_point_density = np.dot(self.point_mass, W_xi_xj)

        assert self.sph_point_density.shape == (self.particles_num, )

        return

    def compute_point_pressure(self):
        '''
            this function will compute the point pressure (not force), according to the formula:
                p = k(ρ-ρ0). k is the gas constant
            now, ρ0 = min(ρ) / 500
        :return: None
        '''
        # rho_0 = np.min(self.sph_point_density) / 500
        rho_0 = 0
        # print(self.sph_point_density)
        self.sph_point_pressure = self.gas_constant * (self.sph_point_density - rho_0)
        assert self.sph_point_pressure.shape == (self.particles_num, )
        return

    def W_poly6_0_order_constant(self, pos):
        '''
            Function: W_poly6
                this function is used to compute the value of kernel poly6:

                                315 / (64 * pi * d^2) * (d^2 - r^2)^3, 0<=r<=d
                W_poly6(r) =
                                0, othersize
        '''
        assert pos.shape == (self.simulator_base_dimension, self.particles_num)
        radius = np.linalg.norm(pos, ord = 2, axis = 0) # radius means r
        radius_2 = radius ** 2
        d_2 = self.kernel_poly6_d ** 2

        # compute the W_poly6(r)
        W_poly6 = self.kernel_poly6_coeff * np.array([ (d_2 - radius_2[i]) ** 3 if radius[i]<= self.kernel_poly6_d else 0 for i in range(self.particles_num)])
        assert  W_poly6.shape == (self.particles_num, )

        # return
        return W_poly6

    def W_poly6_1_order_gradient(self, pos):
        '''
            this function will compute
            2d case:
                dW_poly6_dxy =
                ∇W(|xi - xj|) = (∂x_1, ∂y_1
                                    ...
                                ∂x_j, ∂y_j
                                    ...
                                ∂x_n, ∂y_n))
                               = (particles_num, 2)
                r^2 = x^2 + y^2
                ∂x_j = ∂W/∂x_j = -6 * (d^2 - r^2)^2 * x_j
                ∂y_j = ∂W/∂y_j = -6 * (d^2 - r^2)^2 * y_j
            3d case:
                dW_poly6_dxyz =
                ∇W(|xi - xj|) = (∂x_1, ∂y_1, ∂z_1
                                    ...
                                ∂x_j, ∂y_j, ∂z_j
                                    ...
                                ∂x_n, ∂y_n, ∂z_n))
                               = (particles_num, 3)
                r^2 = x^2 + y^2 + z^2
                ∂x_j = ∂W/∂x_j = -6 * (d^2 - r^2)^2 * x_j
                ∂y_j = ∂W/∂y_j = -6 * (d^2 - r^2)^2 * y_j
                ∂z_j = ∂W/∂y_j = -6 * (d^2 - r^2)^2 * z_j
        :param pos:
        :return:
        '''
        # compute ∂x_j ∂y_j seperately for easy reading
        if self.simulator_base_dimension == 2:
            assert pos.shape[0] == 2
            cur_particle_num = pos.shape[1]
            radius = np.linalg.norm(pos, axis = 0, ord = 2)

            # t1 = time.time()
            # dW_dxy = self.kernel_poly6_coeff * np.array([-6 * ((self.kernel_poly6_d**2 - radius[i] ** 2)**2) * np.reshape(pos[:, i], (2,)) if 0<= radius[i]<= self.kernel_poly6_d else np.zeros(2) for i in range(self.particles_num)]).transpose()
            #t2 = time.time()
            coeff_vector = (self.kernel_poly6_d ** 2 - radius ** 2)**2# d**2 - radius[i]^2yield

            dW_dx = np.array([pos[0, i] if radius[i]< self.kernel_poly6_d else 0 for i in range(cur_particle_num)])
            dW_dy = np.array([pos[1, i] if radius[i]< self.kernel_poly6_d else 0 for i in range(cur_particle_num)])
            dW_dx = np.multiply(coeff_vector, dW_dx)
            dW_dy = np.multiply(coeff_vector, dW_dy)
            assert  dW_dx.shape == (cur_particle_num, )
            assert  dW_dy.shape == (cur_particle_num, )

            # shape the final result dW_dxy, then return
            dW_dxy = np.zeros([2, cur_particle_num])
            dW_dxy[0, :] = dW_dx
            dW_dxy[1, :] = dW_dy

            dW_dxy = -6 * self.kernel_poly6_coeff * dW_dxy
            return dW_dxy
        elif self.simulator_base_dimension == 3:
            assert pos.shape[0] == 3
            cur_particle_num = pos.shape[1]
            radius = np.linalg.norm(pos, axis=0, ord=2)

            # t1 = time.time()
            # dW_dxy = self.kernel_poly6_coeff * np.array([-6 * ((self.kernel_poly6_d**2 - radius[i] ** 2)**2) * np.reshape(pos[:, i], (2,)) if 0<= radius[i]<= self.kernel_poly6_d else np.zeros(2) for i in range(self.particles_num)]).transpose()
            # t2 = time.time()
            coeff_vector = (self.kernel_poly6_d ** 2 - radius ** 2) ** 2  # d**2 - radius[i]^2yield

            dW_dx = np.array([pos[0, i] if radius[i] < self.kernel_poly6_d else 0 for i in range(cur_particle_num)])
            dW_dy = np.array([pos[1, i] if radius[i] < self.kernel_poly6_d else 0 for i in range(cur_particle_num)])
            dW_dz = np.array([pos[2, i] if radius[i] < self.kernel_poly6_d else 0 for i in range(cur_particle_num)])
            dW_dx = np.multiply(coeff_vector, dW_dx)
            dW_dy = np.multiply(coeff_vector, dW_dy)
            dW_dz = np.multiply(coeff_vector, dW_dz)
            assert dW_dx.shape == (cur_particle_num,)
            assert dW_dy.shape == (cur_particle_num,)
            assert dW_dz.shape == (cur_particle_num,)

            # shape the final result dW_dxyz, then return
            dW_dxyz = np.zeros([3, cur_particle_num])
            dW_dxyz[0, :] = dW_dx
            dW_dxyz[1, :] = dW_dy
            dW_dxyz[2, :] = dW_dz

            dW_dxyz = -6 * self.kernel_poly6_coeff * dW_dxyz
            return dW_dxyz

    def W_poly6_2_order_jacob(self, pos_diff):
        if self.simulator_base_dimension == 2:
            ddW_dxy2 = np.zeros([2, self.particles_num * 2]) # shape = (2, 2n)
            '''
                ddW_dxy2_i =    [ ∂^2W/∂x^2  , ∂^2W/∂x∂y, ]     = (2, 2)
                                [ ∂^2W/∂y∂x  , ∂^2W/∂y^2, ]_i
                                
                ddW_dxy2 = [ddW_dxy2_0, ..., ddW_dxy2_j, ..., ddW_dxy2_n] = (2, 2n)
                
                for more details and more formulas, please take a look in the note...
            '''
            radius = np.linalg.norm(pos_diff, ord=2, axis=0)
            assert  radius.shape == (self.particles_num, )

            d2_r2_diff = self.kernel_poly6_d ** 2 - radius ** 2
            assert  d2_r2_diff.shape == (self.particles_num, )

            pos_xx = pos_diff[0] ** 2
            pos_yy = pos_diff[1] ** 2
            pos_xy = np.multiply(pos_diff[0], pos_diff[1])
            for i in range(self.particles_num):
                if radius[i] <= self.kernel_poly6_d:
                    ddW_dxy2[:, 2*i:2*(i+1)] = (6 * d2_r2_diff[i]) * (4 * np.array([[pos_xx[i], pos_xy[i]], [pos_xy[i], pos_yy[i]]]) - d2_r2_diff[i] * np.identity(2))

            # print(ddW_dxy2.shape)
            ddW_dxy2 *= self.kernel_poly6_coeff

            return ddW_dxy2
        elif self.simulator_base_dimension == 3:
            ddW_dxyz2 = np.zeros([3, self.particles_num * 3])  # shape = (3, 3n)
            '''
                ddW_dxyz2_i =    [ ∂^2W/∂x^2  , ∂^2W/∂x∂y, ∂^2W/∂x∂z]     = (3, 3)
                                 [ ∂^2W/∂y∂x  , ∂^2W/∂y^2, ∂^2W/∂y∂z]
                                 [ ∂^2W/∂z∂x  , ∂^2W/∂z∂y, ∂^2W/∂z^2]_i

                ddW_dxyz2 = [ddW_dxyz2_0, ..., ddW_dxyz2_j, ..., ddW_dxyz2_n] = (3, 3n)

                for more details and more formulas, please take a look in the note...
            '''
            radius = np.linalg.norm(pos_diff, ord=2, axis=0)
            assert radius.shape == (self.particles_num,)

            d2_r2_diff = self.kernel_poly6_d ** 2 - radius ** 2
            assert d2_r2_diff.shape == (self.particles_num,)

            pos_xx = pos_diff[0] ** 2
            pos_yy = pos_diff[1] ** 2
            pos_zz = pos_diff[2] ** 2
            pos_xy = np.multiply(pos_diff[0], pos_diff[1])
            pos_xz = np.multiply(pos_diff[0], pos_diff[1])
            pos_yz = np.multiply(pos_diff[0], pos_diff[1])
            for i in range(self.particles_num):
                if radius[i] <= self.kernel_poly6_d:
                    ddW_dxyz2[:, 3 * i:3 * (i + 1)] = (6 * d2_r2_diff[i]) * (
                                4 * np.array([[pos_xx[i], pos_xy[i], pos_xz[i]], [pos_xy[i], pos_yy[i], pos_yz[i]], [pos_xz[i], pos_yz[i], pos_zz[i]]]) - d2_r2_diff[
                            i] * np.identity(3))

            # print(ddW_dxy2.shape)
            ddW_dxyz2 *= self.kernel_poly6_coeff

            return ddW_dxyz2
        else:
            assert 0 == 1