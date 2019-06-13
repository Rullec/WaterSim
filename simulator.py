import numpy as np
import time
'''
    Class ParticleWaterSimulator
        this class is used to simulator a particle system

    simulation formulas(Explicit Euler):
    q means:        point position (2*1)
    q_vel means:    point velocity (2*1)
    q_acc means:    point acceleration  (2*1)
    
    "DOTIMESTEP" IS THE CORE FUNCTION FOR SIMULATION PROCEDURE.
'''

class ParticleWaterSimulatorBase2D:
    # simulation params
    cur_time = 0.0
    timestep = 0.0
    frameid = 0
    particles_num = -1
    space_left_down_corner = (0.0, 0.0)
    space_right_up_corner = (1.0, 1.0)
    g = 0
    collision_detect = False

    # collision penalty force coeff
    collision_epsilon = 0.2
    collision_penalty_k = 3e3  # control the distance
    collision_penalty_b = 1  # control the velocity

    # damping coeff
    damping_coeff = 2

    # status varibles
    point_pos = np.zeros([2, 0])
    point_vel = np.zeros([2, 0])
    point_acc = np.zeros([2, 0])
    point_mass = np.zeros(0)

    # system cost time record
    time_cost_dotimestep = 0.0
    time_cost_compute_force = 0.0
    time_cost_collision_test = 0.0

    def __init__(self,
                 particle_nums_,
                 timestep_,
                 space_left_down_corner_,
                 space_right_up_corner_,
                 gravity_,
                 collision_detect_):
        '''

        :param particle_nums_:
        :param timestep_:
        :param space_left_down_corner_:
        :param space_right_up_corner_:
        :param gravity_:
        '''
        # print('******************Simulator Init Begin****************')

        # system property
        self.particles_num = particle_nums_
        self.point_pos = np.zeros([2, self.particles_num])
        self.point_vel = np.zeros([2, self.particles_num])
        self.point_acc = np.zeros([2, self.particles_num])
        self.point_mass = np.ones(self.particles_num)
        self.space_left_down_corner = space_left_down_corner_
        self.space_right_up_corner = space_right_up_corner_

        # simulation property
        self.g = gravity_
        self.timestep = timestep_

        # init these points
        space_length = space_right_up_corner_[0] - space_left_down_corner_[0]
        space_height = space_right_up_corner_[1] - space_left_down_corner_[1]
        self.point_pos[0, :] = space_length / 2 * np.random.rand(1, self.particles_num) + space_left_down_corner_[
            0] + space_length / 4
        self.point_pos[1, :] = space_height / 2 * np.random.rand(1, self.particles_num) + space_left_down_corner_[
            1] + space_height / 4

        # print('particle_num = %d' % self.particles_num)
        # print('timestep = %d' % self.timestep)
        # print('active space = (%f, %f) - (%f, %f)' % (
        #     space_left_down_corner_[0], space_left_down_corner_[1], self.space_right_up_corner[0],
        #     self.space_right_up_corner[1]))

        # print('******************Simulator Init Succ****************')
        print('[log][simulatorBase] init succ')
        return

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

    # update simulation state
    # currently, the explicit euler method
    def update_state(self):
        '''
            q_vel = q_vel + timestep * q_accel
            q_pos = q_pos + timestep * q_accel
        '''
        self.point_vel += self.timestep * self.point_acc
        self.point_pos += self.timestep * self.point_vel

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
        st = time.time()
        collision_force = np.zeros([2, self.particles_num])
        x_left = 0.0
        x_right = 10.0
        y_down = 0.0
        y_up = 10.0

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

    def compute_damping_force(self):
        return -1 * self.point_vel * self.damping_coeff

    def compute_gravity_force(self):
        assert self.g >0
        assert self.g > 0

        points_gravity = np.zeros([2, self.particles_num])
        points_gravity[1,] = -1 * self.g * self.point_mass

        return points_gravity

class ParticleWaterSimulatorEasy(ParticleWaterSimulatorBase2D):

    # lennard jones forces coef
    lennard_jones_k1 = 0.01
    lennard_jones_k2 = 0.01
    lennard_jones_m = 4
    lennard_jones_n = 2

    def __init__(self, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_,
                 collision_detect_):
        ParticleWaterSimulatorBase2D.__init__(self, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_,
                 collision_detect_)
        return

    def dotimestep(self):
        # dynamic simulation - compute total forces and acceleration
        st = time.time()
        points_force = self.compute_forces()

        # compute accel
        self.point_acc[0,] = points_force[0,] * (1.0 / self.point_mass)
        self.point_acc[1,] = points_force[1,] * (1.0 / self.point_mass)

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
        jones_force = np.zeros([2, self.particles_num])
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
        points_force = np.zeros([2, self.particles_num])

        ## gravity computation
        points_gravity = self.compute_gravity_force()

        # jone forces computation
        points_jone_forces = self.compute_lennard_jones_force()

        # damping
        points_damping_forces = self.compute_damping_force()

        # total force computation
        points_force += points_gravity
        points_force += points_jone_forces
        points_force += points_damping_forces

        # do collision detect and add penalty force
        collision_force = self.do_collision_test_between_wall_and_particles()
        points_force += collision_force
        # print(collision_force[1,])
        self.time_cost_compute_force = time.time() - st

        # print(points_damping_forces)
        return points_force


class ParticleWaterSimulatorSPH(ParticleWaterSimulatorBase2D):

    # kernel parameters
    kernel_poly6_d = 1
    kernel_poly6_coeff = -1

    # sph variables
    sph_kernel_0_order_constact = -1
    sph_kernel_1_order_gradient = -1
    sph_kernel_2_order_jacob = -1
    sph_point_density = -1
    sph_pressure_not_force = -1

    # constant
    gas_constant = 8.314

    def __init__(self, particle_nums_, timestep_, space_left_down_corner_, \
                 space_right_up_corner_, gravity_,collision_detect_):

        ParticleWaterSimulatorBase2D.__init__(particle_nums_, timestep_, space_left_down_corner_, \
            space_right_up_corner_, gravity_,collision_detect_)


        # init sph variable
        self.kernel_poly6_coeff = 315.0 / (64.0 * np.pi * (self.kernel_poly6_d ** 9))

        return

    def dotimestep(self):
        points_force = self.compute_force()

        return

    def compute_force(self):
        sum_force = np.zeros([2, self.particles_num])


        return sum_force

    def compute_W_poly_kernels(self):
        '''
            this function will update W poly variables, including:
                sph_kernel_0_order_constact
                sph_kernel_1_order_gradient
                sph_kernel_2_order_jacob
        :return:
        '''
        self.sph_kernel_0_order_constact = np.zeros([self.particles_num, 1])
        self.sph_kernel_1_order_gradient = np.zeros([self.particles_num, 2])
        self.sph_kernel_2_order_jacob = np.zeros([self.particles_num, 2, 2])

        for i in range(self.particles_num):
            # compute |x - xj| for each point j, shape a vector "dist" = (self.points_num, 1)
            dist = np.reshape(self.point_pos[:, i], (2, 1)) - self.point_pos
            self.sph_kernel_0_order_constact[i] = self.W_poly6_0_order_constant(dist)
            self.sph_kernel_1_order_gradient[i,:] = self.W_poly6_1_order_gradient(dist)
            self.sph_kernel_2_order_jacob[i,:,:] = self.W_poly6_2_order_jacob(dist)

    def W_poly6_0_order_constant(self, pos):
        '''
            Function: W_poly6
                this function is used to compute the value of kernel poly6:

                                315 / (64 * pi * d^2) * (d^2 - r^2)^3, 0<=r<=d
                W_poly6(r) =
                                0, othersize
        '''
        assert pos.shape == (2, self.particles_num)
        radius = np.linalg.norm(pos, ord = 2, axis = 0)
        value = 0
        if radius < self.kernel_poly6_d:
            value = self.kernel_poly6_coeff * (self.kernel_poly6_d ** 2 - radius ** 2) ** 3
        return value

    def W_poly6_1_order_gradient(self, pos):
        assert pos.shape == (self.particles_num, 2)
        value = np.zeros([self.particles_num, 2])
        '''
        value = (∂x_1, ∂y_1
                ∂x_2, ∂y_2
                    ...
                ∂x_j, ∂y_j
                    ...
                ∂x_n, ∂y_n))_(particles_num, 2)
        '''
        if pos < self.kernel_poly6_d:
            assert 0 == 1
            # value[:, 0] =
            # value = -6 * self.kernel_poly6_coeff * ((self.kernel_poly6_d ** 2 - radius ** 2) ** 2 )* radius
        return value

    def W_poly6_2_order_jacob(self, radius):
        value = 0
        if radius < self.kernel_poly6_d:
            value = -6 * self.kernel_poly6_coeff * (1 - 4 * radius ** 2) * (self.kernel_poly6_d ** 2 - radius ** 2)
        return value

    def sph_compute_pressure_force(self, point_index):
        '''
            Function: sph_compute_pressure
                this function is aimed at computing pressure for ith point, its formula:
                pressure_i = -  Σj mj * (pi + pj) / 2 * pj * ∇W(|xi - xj|) = (2, 1)

        '''


    def sph_compute_point_pressure(self):
        '''
            Function: sph_compute_point_pressure
                this function is used to compute pressure in each point
                p = k * (ρ - ρ0), k is the gas constant
        '''
        density_0 = np.minimum(self.sph_point_density) / 2
        # density_0: it means the environment density, I'm not quite sure whether is should be

        self.sph_pressure_not_force = self.gas_constant * (self.sph_point_density - density_0)
        return

    def sph_compute_point_density(self):
        assert self.point_mass.shape == (self.particles_num, 1)
        self.sph_point_density = np.dot(self.sph_kernel_0_order_constact, self.point_mass)
        return