import numpy as np
from src.ForceMomentObjects import Force, Moment, DistributedLoad
import unittest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.NumericalTools import step_function


class ForceMomentSystem(object):

    def __init__(self, list_of_forces: list = [], list_of_moments: list = [], list_of_distributed_forces: list = []):

        self.__distr = list_of_distributed_forces
        self.__forces = list_of_forces
        self.__moments = list_of_moments

        for distr in self.__distr:
            self.__forces += distr.discretize()

    def get_forces(self):
        return self.__forces

    def add_force(self, force_object):
        self.__forces.append(force_object)

    def remove_force(self, idx):
        self.__forces.pop(idx)

    def get_moments(self):
        return self.__moments

    def add_moment(self, moment_object):
        self.__moments.append(moment_object)

    def remove_moment(self, idx):
        self.__moments.pop(idx)

    def get_resultant_force(self):
        return sum(self.__forces)

    def get_resultant_moment(self):
        return sum(self.__moments)

    def plot_force_vectors(self, ax: Axes3D, sizefactor=0.33):

        for force in self.get_forces():
            direction = force.get_direction()
            u, v, w = direction
            length = np.sqrt(u**2 + v**2 + w**2)
            x, y, z = force.get_position()
            ax.quiver(x, z, y, u, w, v, color='r', length=length*sizefactor, pivot='tip')

    def plot_moment_vectors(self, ax: Axes3D, sizefactor=1.0):

        for moment in self.get_moments():
            direction = moment.get_direction()
            u, v, w = direction
            length = np.sqrt(u**2 + v**2 + w**2)
            x, y, z = moment.get_position()
            ax.quiver(x, z, y, u, w, v, color='g', length=length*sizefactor, pivot='tail')


x1 = 0.172
x2 = 1.211
x3 = 2.591
xa = 0.35

R1z = 146.0e3
R2z = -214.0e3
R3z = 83.0e3
R1y = 7681.0
R2y = 0.0
R3y = 7060.0
Pi  = 112.3e3
Pii = 97.4e3
q   = 5.54e3

distance_dict = {'x1': x1,
                 'x2': x2,
                 'x3': x3,
                 'xa': xa}

force_dict = {'R1': [R1y, R1z],
              'R2': [R2y, R2z],
              'R3': [R3y, R3z],
              'Pi': Pi,
              'Pii': Pii,
              'q': q}


def plot_shear(N, x_vals, forces):

    def Fy(x):
        return -forces['R1'][1]*step_function(x, x_vals['x1']) - \
                forces['Pi']*step_function(x, x_vals['x2']-x_vals['xa']/2) + \
                forces['Pii']*step_function(x, x_vals['x2']+x_vals['xa']/2) - \
                forces['R3'][1]*step_function(x, x_vals['x3'])

    def Fz(x):
        return forces['q']*x - \
               forces['R1'][0]*step_function(x, x_vals['x1']) - \
               forces['R2'][0]*step_function(x, x_vals['x2']) - \
               forces['R3'][0]*step_function(x, x_vals['x3'])

    y_shear = []
    z_shear = []

    xrange = np.linspace(0, 2.661, N)

    for xi in xrange:
        y_shear.append(Fy(xi))
        z_shear.append(Fz(xi))

    ax1 = plt.subplot(211)
    plt.plot(xrange, y_shear)
    plt.title('Shear in Y')
    plt.ylabel('Shear Force [N]')
    plt.grid()

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(xrange, z_shear)
    plt.title('Shear in Z')
    plt.xlabel('x-coordinate [m]')
    plt.ylabel('Shear Force [N]')
    plt.grid()


def plot_moments(N):

    def Mx():
        pass

    def My():
        pass

    def Mz():
        pass

    pass


plot_shear(100, distance_dict, force_dict)
plot_moments(100)

if __name__ == "__main__":
    # list_of_forces = [Force(np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]
    # list_of_moments = [Moment(np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]
    # list_of_distr_forces = [DistributedLoad(10, np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)),
    #                                         np.random.randint(-10, 10, (3, 1)), 10)]
    #
    # system = ForceMomentSystem(list_of_forces, list_of_moments, list_of_distr_forces)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # system.plot_force_vectors(ax)
    # system.plot_moment_vectors(ax)

    class ForceMomentSystemTestCases(unittest.TestCase):

        def setUp(self):

            list_of_forces = [Force(np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]
            list_of_moments = [Moment(np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]
            list_of_distr_forces = [DistributedLoad(10, np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]

            self.system = ForceMomentSystem(list_of_forces, list_of_moments, list_of_distr_forces)

