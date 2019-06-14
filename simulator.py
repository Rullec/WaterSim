import numpy as np
import time
import logging

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
    collision_epsilon = -1
    collision_penalty_k = 3e3  # control the distance
    collision_penalty_b = 1  # control the velocity

    # damping coeff
    damping_coeff = 1

    # status varibles
    point_pos = np.zeros([2, 0])
    point_vel = np.zeros([2, 0])
    point_acc = np.zeros([2, 0])
    point_mass = np.zeros(0)

    # system cost time record
    time_cost_dotimestep = 0.0
    time_cost_compute_force = 0.0
    time_cost_collision_test = 0.0

    # logging module
    logger = None

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
        # logging module init
        log_filename = "./log/" + str(time.strftime("%Y-%m-%d %H%M%S", time.localtime())) + str('.txt')
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s  - %(message)s")
        fh = logging.FileHandler(filename=log_filename)  # ouput log both the the console and the file
        logger = logging.getLogger(self.__class__.__name__)
        logger.addHandler(fh)
        logger.info('[SimulatorBase] Init begin')

        # system property
        self.particles_num = particle_nums_
        self.point_pos = np.zeros([2, self.particles_num])
        self.point_vel = np.zeros([2, self.particles_num])
        self.point_acc = np.zeros([2, self.particles_num])
        self.point_mass = np.ones(self.particles_num)
        self.space_left_down_corner = space_left_down_corner_
        self.space_right_up_corner = space_right_up_corner_

        # init these points
        space_length = space_right_up_corner_[0] - space_left_down_corner_[0]
        space_height = space_right_up_corner_[1] - space_left_down_corner_[1]
        self.point_pos[0, :] = space_length / 2 * np.random.rand(1, self.particles_num) + space_left_down_corner_[
            0] + space_length / 4
        self.point_pos[1, :] = space_height / 2 * np.random.rand(1, self.particles_num) + space_left_down_corner_[
            1] + space_height / 10

        # simulation property
        self.g = gravity_
        self.timestep = timestep_
        self.collision_detect = collision_detect_
        self.collision_epsilon = min(space_height, space_length) / 100



        # print('particle_num = %d' % self.particles_num)
        # print('timestep = %d' % self.timestep)
        # print('active space = (%f, %f) - (%f, %f)' % (
        #     space_left_down_corner_[0], space_left_down_corner_[1], self.space_right_up_corner[0],
        #     self.space_right_up_corner[1]))

        # print('******************Simulator Init Succ****************')
        logger.info('[SimulatorBase] Init succ')
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
        self.logger("")
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

        # collision detect
        if self.collision_detect == True:
            collision_force = self.do_collision_test_between_wall_and_particles()
        else:
            collision_force = np.zeros([2, self.particles_num])

        # total force summary
        points_force += points_gravity
        points_force += points_jone_forces
        points_force += points_damping_forces
        points_force += collision_force

        # print(collision_force[1,])
        self.time_cost_compute_force = time.time() - st

        # print(points_damping_forces)
        return points_force

class ParticleWaterSimulatorSPH(ParticleWaterSimulatorBase2D):

    # kernel parameters
    kernel_poly6_d = -1
    kernel_poly6_coeff = -1

    # sph variables
    sph_point_density = -1
    sph_point_pressure = -1

    # constant
    gas_constant = 8.314

    # simulation variables
    viscosity_coeff = 1e-2 # the viscosity of water is 1e-3

    def __init__(self, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_,
                 collision_detect_, kernel_poly6_d_):
        ParticleWaterSimulatorBase2D.__init__(self, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_,
                 collision_detect_)

        # init sph variable
        self.kernel_poly6_d = kernel_poly6_d_
        self.kernel_poly6_coeff = 315.0 / (64.0 * np.pi * (self.kernel_poly6_d ** 9))

        # print('[log][simulator] ParticleWaterSimulatorSPH init succ')
        return

    def dotimestep(self):
        st = time.time()

        # compute force
        points_force = self.compute_force()

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

        return self.point_pos

    def compute_force(self):
        sum_force = np.zeros([2, self.particles_num])

        # 1.1 compute the point density
        self.compute_point_density()

        # 1.2 compute the pressure (not force) in eache point
        self.compute_point_pressure()

        # 1.3 compute the pressure force for each point
        pressure_force = self.compute_pressure_force()

        # 1.4 compute the viscosity force
        viscosity_force = self.compute_viscosity_force()

        # 1.5 compute the gravity
        gravity = self.compute_gravity_force()

        # 1.6 compute the collision force
        # print('collsion = ' + str(self.collision_detect))
        if self.collision_detect == True:
            collision_force = self.do_collision_test_between_wall_and_particles()
        else:
            collision_force = np.zeros([2, self.particles_num])

        # 1.7 compute the damping force
        damping_force = self.compute_damping_force()

        # 1.6 summary
        sum_force += pressure_force
        sum_force += viscosity_force
        sum_force += gravity
        sum_force += collision_force
        sum_force += damping_force

        return sum_force

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
        viscosity_force = np.zeros([2, self.particles_num])

        for i in range(self.particles_num):
            # compute the velocity_diff_coef_vec
            velocity_i = np.reshape(self.point_vel[:, i], (2,1))
            velocity_diff = self.point_vel - velocity_i  # 2*n
            mass_div_density = self.point_mass / self.sph_point_density
            assert mass_div_density.shape == (self.particles_num, )

            velocity_diff_coef_vec = velocity_diff * mass_div_density   # 2 * n
            velocity_diff_coef_vec = velocity_diff_coef_vec.flatten(order='F')
                # velocity_diff_coef_vec = 2n * 1, flatten按列展开
                # (2i， 2i+1)数据对就是第i个点的x y速度差 * 对应系数

            assert velocity_diff_coef_vec.shape == (self.particles_num * 2, )

            # compute the ∇^2_matrix
            pos_i = np.reshape(self.point_pos[:, i], (2,1))
            pos_diff = pos_i - self.point_pos
            ddW_dxy2 = self.W_poly6_2_order_jacob(pos_diff)

            assert ddW_dxy2.shape == (2, 2 * self.particles_num)

            # compute the result
            viscosity_force[:, i] = self.viscosity_coeff * np.dot(ddW_dxy2 , velocity_diff_coef_vec)

            np.set_printoptions(linewidth=200, floatmode='fixed')
            # print('ddW_dxy = ' + str(ddW_dxy2))
            # print('coef = ' + str(velocity_diff_coef_vec))
        # print('viscosity force = ' + str(viscosity_force))
        return viscosity_force

    def compute_pressure_force(self):
        '''
            Function: compute_pressure
                this function is aimed at computing pressure for ith point, its formula:
                pressure_i = -  Σj mj * (pi + pj) / 2 * pj * ∇W(|xi - xj|) = (2, 1)
                           = -  Σj coeff_j * ∇W(|xi - xj|) = (2, 1)
                           = - np.dot(∇W(|xi - xj|)_{2*n}, coeff_vec_j_{n*1}) = (2, 1)
        '''
        pressure_force = np.zeros([2, self.particles_num])
        for i in range(self.particles_num):
            # compute the coeff vector
            coeff_vector = self.point_mass * (self.sph_point_pressure + self.sph_point_pressure[i]) / (2 * self.sph_point_density)
            assert coeff_vector.shape == (self.particles_num, )

            # compute the ∇W(|xi - xj|)
            # print(type(self.point_pos))
            # print(self.point_pos.shape)
            pos_diff = np.reshape(self.point_pos[:, i], (2,1)) - self.point_pos
            dW_dxy = self.W_poly6_1_order_gradient(pos_diff)

            assert dW_dxy.shape == (2, self.particles_num)
            # if np.linalg.norm(dW_dxy) > 1:
                # print(' %d th point info' % i)
                # print('dW_dxy = ' + str(dW_dxy))
                # print('coeff_vector = ' + str(coeff_vector))
            # compute the pressure_i
            pressure_force_i = -np.dot(dW_dxy, coeff_vector)
            pressure_force[:, i] = pressure_force_i
        # print('***************************')
        # print('pressure force = ' + str(pressure_force))
        return pressure_force

    def compute_point_density(self):
        '''
            this function will compute the point density  "self.sph_point_density" # (it's useful for the computation of 2 forces)
            from the formula:
                ρ(x) = \sum_j mj * W(|x-xj|)
                ρ(xi)_{1*1} = np.dot(self.point_mass_{1*n}, W(|xi-xj|)_{n*1})
                ρ(x)_{1*n} = np.dot(self.point_mass_{1*n}, W(|xi-xj|)_{n*n})
        :return: None
        '''

        # compute W(|xi-xj|_{n*n})
        W_xi_xj = np.zeros([self.particles_num, self.particles_num])
        for i in range(W_xi_xj.shape[1]):
            dist = np.reshape(self.point_pos[:, i], (2, 1)) - self.point_pos
            assert dist.shape == (2, self.particles_num)
            W_xi_xj[:, i] = self.W_poly6_0_order_constant(dist)
        # print(W_xi_xj)

        # compute the point density
        assert self.point_mass.shape == (self.particles_num, )
        self.sph_point_density = np.dot(self.point_mass, W_xi_xj)
        # print('point_mass = %s' % str(self.point_mass.transpose()))
        # print('W_xi_xj = %s' % str(W_xi_xj))
        # print('res = %s' % str(self.sph_point_density))
        assert self.sph_point_density.shape == (self.particles_num, )

        return

    def compute_point_pressure(self):
        '''
            this function will compute the point pressure (not force), according to the formula:
                p = k(ρ-ρ0). k is the gas constant
            now, ρ0 = min(ρ) / 500
        :return: None
        '''
        rho_0 = np.min(self.sph_point_density) / 500
        # rho_0 = 0
        self.sph_point_pressure = self.gas_constant * (self.sph_point_density - rho_0)

        return

    def W_poly6_0_order_constant(self, pos):
        '''
            Function: W_poly6
                this function is used to compute the value of kernel poly6:

                                315 / (64 * pi * d^2) * (d^2 - r^2)^3, 0<=r<=d
                W_poly6(r) =
                                0, othersize
        '''
        assert pos.shape == (2, self.particles_num)
        radius = np.linalg.norm(pos, ord = 2, axis = 0) # radius means r

        # compute the W_poly6(r)
        W_poly6 = [self.kernel_poly6_coeff * (self.kernel_poly6_d ** 2 - radius[i] ** 2) ** 3 if 0<= radius[i]<= self.kernel_poly6_d \
                 else 0 for i in range(self.particles_num)]
        W_poly6 = np.array(W_poly6, )
        assert  W_poly6.shape == (self.particles_num, )

        # return
        return W_poly6

    def W_poly6_1_order_gradient(self, pos):
        '''
            this function will compute
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

        :param pos:
        :return:
        '''
        # compute ∂x_j ∂y_j seperately for easy reading

        assert pos.shape == (2, self.particles_num)
        radius = np.linalg.norm(pos, axis = 0, ord = 2)

        # 这里可以用列表生成式优化
        dW_dx = self.kernel_poly6_coeff * np.array([-6 * ((self.kernel_poly6_d**2 - radius[i] ** 2)**2) * pos[0, i] if 0<= radius[i]<= self.kernel_poly6_d else 0 for i in range(self.particles_num)])
        dW_dy = self.kernel_poly6_coeff * np.array([-6 * ((self.kernel_poly6_d**2 - radius[i] ** 2)**2) * pos[1, i] if 0<= radius[i]<= self.kernel_poly6_d else 0 for i in range(self.particles_num)])
        assert  dW_dx.shape == (self.particles_num, )
        assert  dW_dy.shape == (self.particles_num, )

        # shape the final result dW_dxy, then return
        dW_dxy = np.zeros([2, self.particles_num])
        dW_dxy[0, :] = dW_dx
        dW_dxy[1, :] = dW_dy

        return dW_dxy

    def W_poly6_2_order_jacob(self, pos_diff):
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
            if 0 <= radius[i] <= self.kernel_poly6_d:
                ddW_dxy2[:, 2*i:2*(i+1)] = (6 * d2_r2_diff[i]) * (4 * np.array([[pos_xx[i], pos_xy[i]], [pos_xy[i], pos_yy[i]]]) - d2_r2_diff[i] * np.identity(2))
        ddW_dxy2 *= self.kernel_poly6_coeff

        return ddW_dxy2