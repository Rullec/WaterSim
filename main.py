import matplotlib.pyplot as plt
import numpy as np
import time

'''
    Class ParticleWaterSimulator
        this class is used to simulator a particle system
    
    simulation formulas(Explicit Euler):
    q means:        point position (2*1)
    q_vel means:    point velocity (2*1)
    q_acc means:    point acceleration  (2*1)
'''

class ParticleWaterSimulator:

    # simulation params
    cur_time = 0.0
    timestep = 0.0
    particles_num = -1
    space_left_down_corner = (0.0, 0.0)
    space_right_up_corner = (1.0, 1.0)
    gravity = 0

    # system status varibles
    point_pos = np.zeros([2, 0])
    point_vel = np.zeros([2, 0])
    point_acc = np.zeros([2, 0])


    def __init__(self, particle_nums_, timestep_, space_left_down_corner_, space_right_up_corner_, gravity_):
        '''

        :param particle_nums_:
        :param timestep_:
        :param space_left_down_corner_:
        :param space_right_up_corner_:
        :param gravity_:
        '''
        print('******************Simulator Init Begin****************')

        # system property
        self.particles_num = particle_nums_
        self.point_pos = np.zeros([2, self.particles_num])
        self.point_vel = np.zeros([2, self.particles_num])
        self.point_accel = np.zeros([2, self.particles_num])
        self.space_left_down_corner = space_left_down_corner_
        self.space_right_up_corner = space_right_up_corner_

        # simulation property
        self.gravity = gravity_
        self.timestep = timestep_

        # init these points
        space_length = space_right_up_corner_[0] - space_left_down_corner_[0]
        space_height = space_right_up_corner_[1] - space_left_down_corner_[1]
        self.point_pos[0,:] = space_length * np.random.rand(1, particles_num) + space_left_down_corner_[0]
        self.point_pos[1,:] = space_height * np.random.rand(1, particles_num) + space_left_down_corner_[1]

        print('particle_num = %d' % self.particles_num)
        print('timestep = %d' % self.timestep)
        print('active space = (%f, %f) - (%f, %f)' % (space_left_down_corner_[0], space_left_down_corner_[1], self.space_right_up_corner[0], self.space_right_up_corner[1]))

        print('******************Simulator Init Succ****************')
        return

    def dotimestep(self):
        # dynamic simulation - compute total forces and acceleration
        points_force = np.ones([2, ])

        # dynamic simulator - compute total

        # update the state

        # time ++
        self.cur_time += timestep
        print('[log][simulator] do timestep, cur time = %.3f s' % self.cur_time)
        return self.point_pos

    # get and set methods
    def get_cur_time(self):
        return self.cur_time

particles_num = 1000
timestep = 0.01

# simulator init
sim = ParticleWaterSimulator(
    particle_nums_ = particles_num,
    timestep_= 0.01,
    space_left_down_corner_= (0.0, 0.0),
    space_right_up_corner_ = (1.0, 1.0),
    gravity_= -9.8)

t_last = time.time()
plt.ion()
while True:

    # do timestep
    points = sim.dotimestep()

    # display
    plt.scatter(points[0], points[1])
    plt.title("FPS: %d, particle_num: %d, cur_time = %.2f" % (int(1.0/(time.time() - t_last)), particles_num, sim.get_cur_time()))
    t_last = time.time()
    plt.pause(0.001)
    plt.cla()
