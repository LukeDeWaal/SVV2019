"""
Main Script
"""

from src.ForceMomentObjects import Force, Moment, DistributedLoad
from src.AppliedForcesMoments import ForceMomentSystem
from src.Structure import FullModel, get_crossectional_coordinates
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Main:

    def __init__(self, model, force_moment_system):

        self.__model = model
        self.__force_moment_sys = force_moment_system

        self.__fig = plt.figure()
        self.__ax = Axes3D(self.__fig)

    def __get_axes(self):
        return self.__ax

    def run_simulation(self):
        pass

    def plot_model_and_forces(self):

        self.__model.plot_structure(self.__get_axes())
        self.__force_moment_sys.plot_force_vectors(self.__get_axes())
        self.__force_moment_sys.plot_moment_vectors(self.__get_axes(), sizefactor=1)

    def plot_results(self):
        pass

    def log_results(self):
        pass

    def load_results(self):
        pass

if __name__ == "__main__":
    ha = 0.205
    Ca = 0.605
    pi = np.pi
    hs = 1.6 * 10 ** (-2)

    coordinates = get_crossectional_coordinates(Ca, ha, hs)
    model = FullModel(coordinates, (-1, 1), 10)
    FM_sys = ForceMomentSystem(list_of_moments=[Moment([1000, 0, 0], [0, 0, ha/2])], list_of_distributed_forces=[DistributedLoad(20, [-1, ha/2, ha/2], [1, ha/2, ha/2], [0, -1, -1], 10)])

    main = Main(model, FM_sys)
    main.plot_model_and_forces()
