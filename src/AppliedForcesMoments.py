import numpy as np
from src.ForceMomentObjects import Force, Moment, DistributedLoad
import unittest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


if __name__ == "__main__":

    class ForceMomentSystemTestCases(unittest.TestCase):

        def setUp(self):

            list_of_forces = [Force(np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]
            list_of_moments = [Moment(np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]
            list_of_distr_forces = [DistributedLoad(10, np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)), np.random.randint(-10, 10, (3, 1)))]

            self.system = ForceMomentSystem(list_of_forces, list_of_moments, list_of_distr_forces)

