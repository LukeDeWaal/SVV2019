"""
Build the aileron structural idealization here (using skin, booms, etc)
"""

from src.Idealizations import *
import matplotlib.pyplot as plt
import unittest
import numpy as np

ha = 0.205
Ca = 0.605
pi = np.pi


def plot_aileron(c, h):
    curved_coordinates = []
    straight_coordinates = []

    #curved part
    for theta in np.arange(pi/2+pi/4, 3*pi/2, pi/4):
        curved_coordinates.append((None, np.sin(theta)*h/2, np.cos(theta)*h/2+h/2))

    #straight part
    def lower_slope(x):
        return (h/2)/(c-h/2)*x - h/2

    def upper_slope(x):
        return -(h/2)/(c-h/2)*x + h/2

    for slope in [lower_slope, upper_slope]:
        for x in np.arange(h/2, c, (c-h/2)/6):
            straight_coordinates.append((None, slope(x), x))

    return curved_coordinates+straight_coordinates

a = np.array(plot_aileron(Ca, ha))

plt.scatter(a[:,2], a[:,1])

class CrossSection(object):

    def __init__(self, *objects: "Booms and/or Skin"):

        self.__objects = objects
        self.__structure = None


class Structure(object):

    def __init__(self, *sections):

        pass


if __name__ == "__main__":

    class StructureTestCases(unittest.TestCase):

        def setUp(self):

            self.structure = CrossSection()


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(StructureTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
