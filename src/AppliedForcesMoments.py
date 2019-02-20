import numpy as np
from src.ForceMomentObjects import Force, Moment
import unittest


class ForceMomentSystem(object):

    def __init__(self, list_of_forces: list, list_of_moments: list):

        self.__forces = list_of_forces
        self.__moments = list_of_moments

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


if __name__ == "__main__":

    class ForceMomentSystemTestCases(unittest.TestCase):

        def setUp(self):

            pass

