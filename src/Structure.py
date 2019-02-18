"""
Build the aileron structural idealization here (using skin, booms, etc)
"""

from src.Idealizations import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        return [h/2*np.cos(theta), h/2*(1-np.sin(theta))]

    def lower_slope(x):
        return (h/2)/(c-h/2)*x - h/2

    def upper_slope(x):
        return -(h/2)/(c-h/2)*x + h/2

    # Curved Part Cooprdinates
    for theta in np.arange(pi/4,pi, pi/4):
        curved_coordinates.append(curved_part(theta))

    # Need to check what the maximum distance is untill stiffeneres dont fit anymore
    angle = np.arctan((h/2)/(c-h/2))    # Stiffener angle with vertical
    max_height_diff = hs*np.cos(angle)  # Vertical height of stiffener

    max_x_1 = newtons_method(lambda x: upper_slope(x)-max_height_diff, c)
    max_x_2 = newtons_method(lambda x: lower_slope(x)+max_height_diff, c)
    if round(max_x_1, 8) == round(max_x_2, 8):
        pass
    else:
        return

    # Set the x-coordinates for the straight skin parts
    x_range = np.linspace(h/2, max_x_1, 7)
    for x in x_range[1:]:
        straight_coordinates.append([lower_slope(x), x])

    for x in x_range[1:][::-1]:
        straight_coordinates.append([upper_slope(x), x])

    array = np.array(curved_coordinates+straight_coordinates)   # Merge Coordinate Lists

    return np.concatenate((np.zeros((len(array), 1)), array), axis=1)


def plot_crosssection(positions):

    plt.scatter(positions[:,1], positions[:,2])
    plt.axes().set_aspect('equal')
    plt.grid(True)

#plot_crosssection(get_crossectional_coordinates(Ca, ha, h_stringer))


class CrossSection:

    def __init__(self, coordinates, x_coordinate=0):

        self.__coordinates = coordinates

        # Creating all Boom Objects
        self.__boom_objects = self.__initialize_boom_objects(self.__coordinates)

        # Creating all Skin Objects between the Boom Objects
        self.__skin_objects = self.__initialize_skin_objects(self.__boom_objects)

        self.set_x(x_coordinate)

    @staticmethod
    def __initialize_boom_objects(coordinates):
        return [Boom(1, 1, coordinate) for coordinate in coordinates]

    @staticmethod
    def __initialize_skin_objects(boom_objects):
        skin_objects = []
        i = -1
        while True:
            skin = StraightSkin(1, 1, boom_objects[i].get_position(), boom_objects[i + 1].get_position())
            skin_objects.append(skin)

            if len(skin_objects) == len(boom_objects):
                break

            i += 1

        return skin_objects

    def set_x(self, x):
        for boom in self.get_boom_objects():
            old_position = boom.get_position()
            old_position[0] = x
            boom.set_position(old_position)

    def get_x(self):
        return self.get_boom_objects()[0].get_position()[0]

    def area_MOI(self, axis1, axis2=None):
        pass

    def get_mass(self):
        pass

    def get_boom_objects(self):
        return self.__boom_objects

    def get_skin_objects(self):
        return self.__skin_objects

    def get_objects(self):
        return [self.get_boom_objects(), self.get_skin_objects()]

    def get_coordinates(self):
        return np.array([boom.get_position() for boom in self.get_boom_objects()])

    def calculate_shear_centre(self):
        pass


class FullModel(object):

    def __init__(self, coordinates, xrange, N, *forces_and_moments):

        self.__sections = []
        self.__coordinates = coordinates
        self.__xrange = xrange
        self.__N = N

        self.__forcesandmoments = forces_and_moments

        self.__assemble_structure()

    def __assemble_structure(self):

        for xi in np.linspace(self.__xrange[0], self.__xrange[1], self.__N):
            print(xi)
            self.__sections.append(CrossSection(self.__coordinates, x_coordinate=xi))

    def get_all_coordinates(self):

        coordinates = tuple([section.get_coordinates() for section in self.get_sections()])
        #return np.concatenate(coordinates, axis=0)
        return coordinates

    def calculate_reaction_forces(self):
        pass

    def plot_structure(self):
        pass

    def get_mass(self):
        pass

    def get_sections(self):
        return self.__sections


if __name__ == "__main__":

    boompos = get_crossectional_coordinates(Ca, ha, h_stringer)

    #CS = CrossSection(a, 4)
    model = FullModel(boompos, (-2, 2), 5)
    sections = model.get_sections()


    class StructureTestCases(unittest.TestCase):

        def setUp(self):

            self.ha = 0.205
            self.Ca = 0.605
            self.h_stringer = 1.6 * 10 ** (-2)

            self.coordinates = get_crossectional_coordinates(self.Ca, self.ha, self.h_stringer)
            self.crosssection = CrossSection(self.coordinates)

        def test_boom_coordinate_calculations(self):

            for idx, coordinate in enumerate(self.coordinates):
                self.assertEqual(coordinate[0], 0)

                if idx == 1:
                    self.assertAlmostEqual(coordinate[1], 0.0)

                if 14 >= idx > 3:
                    self.assertGreater(self.coordinates[idx][1], self.coordinates[idx-1][1])

        def test_crosssection_x_coordinate_methods(self):

            self.assertAlmostEqual(self.crosssection.get_x(), 0.0)
            self.crosssection.set_x(4.0)
            self.assertEqual(self.crosssection.get_x(), 4.0)

            for boom in self.crosssection.get_boom_objects():
                self.assertEqual(boom.get_position()[0], 4.0)


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(StructureTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
