"""
Build the aileron structural idealization here (using skin, booms, etc)
"""

from src.Idealizations import *
import matplotlib.pyplot as plt
import unittest
import numpy as np
from src.NumericalTools import newtons_method, derive

ha = 0.205
Ca = 0.605
pi = np.pi
h_stringer = 1.6*10**(-2)


def get_crossectional_coordinates(c, h, hs):
    """
    Function to calculate a possible solution for evenly spaced stiffeners
    :param c: Chordlength
    :param h: Height
    :param hs: Stiffener Height
    :return: Array of Coordinates
    """
    curved_coordinates = []
    straight_coordinates = []

    def curved_part(theta):
        return [h/2*(1-np.sin(theta)), h/2*np.cos(theta)]

    def lower_slope(x):
        return (h/2)/(c-h/2)*x - h/2

    def upper_slope(x):
        return -(h/2)/(c-h/2)*x + h/2

    #Curved Part Cooprdinates
    for theta in np.arange(pi/4,pi, pi/4):
        curved_coordinates.append(curved_part(theta))

    #Need to check what the maximum distance is untill stiffeneres dont fit anymore
    angle = np.arctan((h/2)/(c-h/2))   #Stiffener angle with vertical
    max_height_diff = hs*np.cos(angle) #Vertical height of stiffener

    max_x_1 = newtons_method(lambda x: upper_slope(x)-max_height_diff, c)
    max_x_2 = newtons_method(lambda x: lower_slope(x)+max_height_diff, c)
    if round(max_x_1, 8) == round(max_x_2, 8):
        print(max_x_1)
    else:
        return

    x_range = np.linspace(h/2, max_x_1, 7)
    for x in x_range[1:]:
        straight_coordinates.append([x, lower_slope(x)])

    for x in x_range[1:][::-1]:
        straight_coordinates.append([x, upper_slope(x)])

    array = np.array(curved_coordinates+straight_coordinates)

    return np.concatenate((np.zeros((len(array), 1)), array), axis=1)


def plot_crosssection(positions):

    plt.scatter(positions[:,0], positions[:,1])
    plt.axes().set_aspect('equal')
    plt.grid(True)


class CrossSection(object):

    def __init__(self, coordinates, x_coordinate=0):

        self.__coordinates = coordinates
        self.__coordinates[:,0] += x_coordinate

    def area_MOI(self, axis1, axis2=None):
        pass

    def get_mass(self):
        pass

    def get_objects(self):
        pass

    def calculate_shear_centre(self):
        pass


class FullModel(object):

    def __init__(self, *crosssections, **forcesandmoments):
        pass

    def assemble_structure(self):
        pass

    def calculate_reaction_forces(self):
        pass

    def plot_structure(self):
        pass

    def get_mass(self):
        pass

    def get_objects(self):
        pass

if __name__ == "__main__":

    class StructureTestCases(unittest.TestCase):

        def setUp(self):
        pass


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(StructureTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
