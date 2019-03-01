import numpy as np
from ForceMomentObjects import Force, Moment, DistributedLoad
#from src.ForceMomentObjects import Force, Moment, DistributedLoad
import unittest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from src.NumericalTools import step_function, reLu, integrate
from NumericalTools import step_function, reLu, integrate

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



Pi  = 104.4114e3
Pii = 97.4e3
q   = 5.54e3

distance_dict = {'x1': 0.172,
                 'x2': 1.211,
                 'x3': 2.591,
                 'xa': 0.35,
                 'theta': 28.0*np.pi/180.0,
                 'ba': 2.661,
                 'ha': 0.205,
                 'Ca': 0.605
                }

force_dict = {'R1':  [0, -34990.32, 117433.67],
              'R2':  [0, 57917.31, -170740.58],
              'R3':  [0, -13202.29, 66418.53],
              'Pi':  [0, Pi*np.sin(distance_dict['theta']), Pi*np.cos(distance_dict['theta'])],
              'Pii': [0, Pii*np.sin(distance_dict['theta']), Pii*np.cos(distance_dict['theta'])],
              'q':   [0, q*np.cos(distance_dict['theta']), q*np.sin(distance_dict['theta'])]}


def moment_functions(x_vals, forces):

    def Mx(x):

        return (+ forces['q'][1] * x *abs(x_vals['Ca']/4-x_vals['ha']/2)
                - forces['Pi'][2] * x_vals['ha']/2 * step_function(x,(x_vals['x2']-x_vals['xa']/2))
                + forces['Pi'][1] * x_vals['ha'] / 2 * step_function(x, (x_vals['x2'] - x_vals['xa'] / 2))
                - forces['Pii'][1] * x_vals['ha'] / 2 * step_function(x, (x_vals['x2'] + x_vals['xa'] / 2))
                + forces['Pii'][2] * x_vals['ha'] / 2 * step_function(x, (x_vals['x2'] + x_vals['xa'] / 2)))*-1

    def My(x):

        return (+ forces['q'][2]*(x**2)/2
                - forces['R1'][2]*reLu(x,x_vals['x1'])
                + forces['Pi'][2]*reLu(x,(x_vals['x2']-x_vals['xa']/2))
                - forces['R2'][2] * reLu(x, x_vals['x2'])
                - forces['Pii'][2]*reLu(x, (x_vals['x2']+x_vals['xa']/2))
                - forces['R3'][2]*reLu(x, x_vals['x3']))*-1

    def Mz(x):

        return (+forces['q'][1]*(x**2)/2
                - forces['R1'][1]*reLu(x,x_vals['x1'])
                - forces['Pi'][1]*reLu(x,(x_vals['x2']-x_vals['xa']/2))
                - forces['R2'][1]*reLu(x, x_vals['x2'])
                + forces['Pii'][1]*reLu(x, (x_vals['x2']+x_vals['xa']/2))
                - forces['R3'][1]*reLu(x,x_vals['x3']))

    return Mx, My, Mz


def shear_functions(x_vals, forces):

    def Fy(x):
        return (- forces['q'][1]  * x \
               + forces['R1'][1] * step_function(x, x_vals['x1'])\
               + forces['R3'][1] * step_function(x, x_vals['x3'])\
               + forces['R2'][1] * step_function(x, x_vals['x2'])\
               + forces['Pi'][1] * step_function(x, x_vals['x2']-x_vals['xa']/2.0) \
               - forces['Pii'][1]* step_function(x, x_vals['x2']+x_vals['xa']/2.0))*-1

    def Fz(x):
        return (+ forces['q'][2]  * x \
               - forces['R1'][2] * step_function(x, x_vals['x1']) \
               - forces['R2'][2] * step_function(x, x_vals['x2']) \
               - forces['R3'][2] * step_function(x, x_vals['x3']) \
               + forces['Pi'][2] * step_function(x, (x_vals['x2'] - x_vals['xa']/2)) \
               - forces['Pii'][2]* step_function(x, (x_vals['x2'] + x_vals['xa']/2)))*-1

    return Fy, Fz


def plot_shear(N, x_vals, forces):

    Fy, Fz = shear_functions(x_vals, forces)

    y_shear = []
    z_shear = []

    xrange = np.linspace(0, 2.661, N)

    for xi in xrange:
        y_shear.append(Fy(xi))
        z_shear.append(Fz(xi))

    fig = plt.figure()

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
    plt.show()


def plot_moments(N, x_vals, forces):

    Mx, My, Mz = moment_functions(x_vals, forces)

    x_moments = []
    y_moments = []
    z_moments = []

    xrange = np.linspace(0, 2.661, N)

    for xi in xrange:
        x_moments.append(Mx(xi))
        y_moments.append(My(xi))
        z_moments.append(Mz(xi))

    fig = plt.figure()

    ax1 = plt.subplot(311)
    plt.plot(xrange, x_moments)
    plt.title('Moments in X')
    plt.ylabel('Moment [N*m]')
    plt.grid()

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(xrange, y_moments)
    plt.title('Moments in Y')
    plt.ylabel('Moment [N*m]')
    plt.grid()

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(xrange, z_moments)
    plt.title('Moments in Z')
    plt.ylabel('Moment [N*m]')
    plt.grid()
    plt.show()





# plot_shear(1000, distance_dict, force_dict)
# plot_moments(1000, distance_dict, force_dict)
#plot_displacements(1000, distance_dict, force_dict)

plot_shear(1000, distance_dict, force_dict)
plot_moments(1000, distance_dict, force_dict)
# plot_displacements(1000, distance_dict, force_dict)


if __name__ == "__main__":

    class ForceMomentSystemTestCases(unittest.TestCase):

        def setUp(self):

            list_of_forces = [Force(np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]
            list_of_moments = [Moment(np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]
            list_of_distr_forces = [DistributedLoad(10, np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]

            self.system = ForceMomentSystem(list_of_forces, list_of_moments, list_of_distr_forces)

