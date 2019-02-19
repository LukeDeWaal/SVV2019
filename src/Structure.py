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


def get_crossectional_coordinates(c, h, hs) -> np.array:
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
        return (h/2)/(c-h/2)*(x-h/2) - h/2

    def upper_slope(x):
        return -(h/2)/(c-h/2)*(x-h/2) + h/2

    # Curved Part Cooprdinates
    for theta in np.arange(pi/4, pi, pi/4):
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

    return array


def plot_crosssection(positions) -> None:

    plt.scatter(positions[:,1], positions[:,2])
    plt.axes().set_aspect('equal')
    plt.grid(True)

#plot_crosssection(get_crossectional_coordinates(Ca, ha, h_stringer))


class CrossSection:

    def __init__(self, boom_coordinates, sparcap_coordinates=[np.array([ha/2, ha/2]), np.array([-ha/2, ha/2])], x_coordinate=0):

        self.__cur = 0

        self.__x = x_coordinate
        self.__x_vector = np.ones((len(boom_coordinates[:, 0]), 1)) * self.__x

        self.__stiffener_coordinates    = np.concatenate((self.__x_vector, boom_coordinates), axis=1)
        self.__sparcap_coordinates = np.concatenate((np.ones((2, 1)) * self.__x, sparcap_coordinates), axis=1)

        # Creating all Boom Objects
        self.__stiffener_booms = self.__initialize_boom_objects(self.__stiffener_coordinates)
        self.__spar_booms = self.__initialize_spar_caps(self.__sparcap_coordinates)
        self.__all_booms = self.__stiffener_booms
        self.__all_booms.insert(0, self.__spar_booms[0])
        self.__all_booms.insert(4, self.__spar_booms[1])

        # Creating all Skin Objects between the Boom Objects
        self.__skin_objects = self.__initialize_skin_objects(self.__all_booms)

        # Calculate initial centroid
        self.__centroid = self.get_centroid()

        # Calculate amount of booms
        self.__N_booms = len(self.get_boom_objects())
        self.__N_spars = len(self.get_spar_caps())
        self.__N_total = self.__N_booms + self.__N_spars

    @staticmethod
    def __initialize_boom_objects(coordinates):
        return [Boom(1, 1, coordinate) for coordinate in coordinates]

    @staticmethod
    def __initialize_spar_caps(coordinates):
        return [Boom(1, 2, coordinate) for coordinate in coordinates]

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

    @staticmethod
    def __parametrize_line(v1: np.array, v2: np.array):
        def line(t):
            return v1 + (v2-v1)*t
        return line

    def __str__(self):
        return f"# Booms: {self.get_N_booms()}, # Spar Caps: {self.get_N_spars()} \n"

    def __len__(self):
        return len(self.get_skin_objects())+len(self.get_all_booms())

    def __iter__(self):
        self.__cur = 0
        return self

    def __next__(self):
        if self.__cur >= len(self.get_all_booms()):
            raise StopIteration()

        result = self.get_all_booms()[self.__cur]
        self.__cur += 1
        return result

    def get_N_booms(self):
        return self.__N_booms

    def get_N_spars(self):
        return self.__N_spars

    def get_N_objects(self):
        return self.get_N_booms() + self.get_N_spars()

    def set_x(self, x):
        """
        Set the x-coordinate of the crosssection
        :param x: X Coordinate
        :return: None
        """
        for boom in self.get_boom_objects():
            old_position = boom.get_position()
            old_position[0] = x
            boom.set_position(old_position)

    def get_x(self):
        """
        Check the x-value of the crosssection
        :return: x coordinate
        """
        return self.get_boom_objects()[0].get_position()[0]

    def get_centroid(self):
        """
        Calculate the centroid of the crosssection
        :return: [x, y_bar, z_bar]
        """
        ybar_top = sum([boom.get_size()*boom.get_position()[1] for boom in self.get_all_booms()])
        ybar_bot = sum([boom.get_size() for boom in self.get_all_booms()])

        ybar = ybar_top/ybar_bot

        zbar_top = sum([boom.get_size() * boom.get_position()[2] for boom in self.get_all_booms()])
        zbar_bot = sum([boom.get_size() for boom in self.get_all_booms()])

        zbar = zbar_top/zbar_bot

        return np.array([self.__x, ybar, zbar])

    def area_MOI(self, axes: str):
        """
        Calculate the MOI with the defined axes
        :param axes: 'z', 'zz', 'y', 'yy', 'zy'
        :return: MOI around defined axes
        """
        MOI = 0

        if axes == ('z' or 'zz'):
            idx1 = idx2 = 1

        elif axes == ('y' or 'yy'):
            idx1 = idx2 = 2

        elif axes == ('zy' or 'yz'):
            idx1 = 1
            idx2 = 2

        else:
            return -1

        for boom in self.get_all_booms():
            MOI += boom.get_size() * (boom.get_position()[idx1] * boom.get_position()[idx2])
        return MOI

    def update_boom_area(self, idx):
        """
        :param idx: Index of boom item (top sparcap is idx=0, then moves counterclockwise)
        :return: None
        """
        pass

    def get_mass(self):
        """
        Get total summed up mass of booms and skins
        :return: Total Mass
        """
        return sum([skin.get_mass() for skin in self.get_skin_objects()]) + \
               sum([boom.get_mass() for boom in self.get_all_booms()])

    def get_spar_caps(self):
        """
        :return: Spar Cap Objects
        """
        return self.__spar_booms

    def get_all_booms(self):
        return self.__all_booms

    def get_boom_objects(self):
        return self.__stiffener_booms

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

        self.__sections = [None]*N
        self.__cur = 0
        self.__boomcoordinates = coordinates
        self.__xrange = xrange
        self.__N = N

        self.__forcesandmoments = forces_and_moments

        self.__assemble_structure()

    def __len__(self):
        return self.__N

    def __str__(self):
        string = ""
        for section in self:
            string += str(section)

    def __iter__(self):
        self.__cur = 0
        return self

    def __next__(self):

        if self.__cur >= self.__N + 1:
            raise StopIteration()
        else:
            result = self.__sections[self.__cur]
            return result

    def __assemble_structure(self):

        for idx, xi in enumerate(np.linspace(self.__xrange[0], self.__xrange[1], self.__N)):
            self.__sections[idx] = CrossSection(self.__boomcoordinates, x_coordinate=xi)

    def get_all_coordinates(self):

        coordinates = tuple([section.get_coordinates() for section in self.get_sections()])
        return np.concatenate(coordinates, axis=0)
        #return coordinates

    def calculate_reaction_forces(self):
        pass

    @staticmethod
    def __parametrize_line(v1: np.array, v2: np.array):
        def line(t):
            return v1 + (v2-v1)*t
        return line

    def plot_structure(self):

        fig = plt.figure()
        ax = Axes3D(fig)

        # Boom Coordinates
        coordinates = self.get_all_coordinates()
        xboomplot = coordinates[:, 0]
        yboomplot = coordinates[:, 1]
        zboomplot = coordinates[:, 2]

        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-0.1, 0.6)
        ax.set_zlim3d(-0.3, 0.3)

        # Lines around crosssection
        # xlineplot_1 = [[] for _ in range(self.__N)]
        # ylineplot_1 = [[] for _ in range(self.__N)]
        # zlineplot_1 = [[] for _ in range(self.__N)]
        #
        # for idx, section in enumerate(self.get_sections()):
        #     section_coordinates = section.get_all_booms()
        #     for boom in section_coordinates:
        #         position = boom.get_position()
        #         xlineplot_1[idx].append(position[0])
        #         ylineplot_1[idx].append(position[1])
        #         zlineplot_1[idx].append(position[2])
        #     xlineplot_1[idx].append(section_coordinates[0].get_position()[0])
        #     ylineplot_1[idx].append(section_coordinates[0].get_position()[1])
        #     zlineplot_1[idx].append(section_coordinates[0].get_position()[2])
        #
        #
        #
        # # Lines through crosssection
        # xlineplot_2 = [[] for _ in range(17)]
        # ylineplot_2 = [[] for _ in range(17)]
        # zlineplot_2 = [[] for _ in range(17)]
        #
        # for section in self.get_sections():
        #     section_coordinates = section.get_all_booms()
        #     for idx, boom in enumerate(section_coordinates):
        #         position = boom.get_position()
        #         xlineplot_2[idx].append(position[0])
        #         ylineplot_2[idx].append(position[1])
        #         zlineplot_2[idx].append(position[2])

        ax.scatter3D(xboomplot, zboomplot, yboomplot, s=40, c='k')
        #
        # for i in range(self.__N):
        #     ax.plot(xlineplot_1[i], zlineplot_1[i], ylineplot_1[i], 'r')
        #
        # for i in range(17):
        #     ax.plot(xlineplot_2[i], zlineplot_2[i], ylineplot_2[i], 'r')

    def get_mass(self):
        pass

    def get_sections(self):
        return self.__sections


if __name__ == "__main__":

    boompos = get_crossectional_coordinates(Ca, ha, h_stringer)

    #CS = CrossSection(a, 4)
    model = FullModel(boompos, (-2.661/2, 2.661/2), 10)
    a = model.get_all_coordinates()
    model.plot_structure()

    class StructureTestCases(unittest.TestCase):

        def setUp(self):

            self.ha = 0.205
            self.Ca = 0.605
            self.h_stringer = 1.6 * 10 ** (-2)

            self.coordinates = get_crossectional_coordinates(self.Ca, self.ha, self.h_stringer)
            self.crosssection = CrossSection(self.coordinates)


        def test_boom_coordinate_calculations(self):

            for idx, coordinate in enumerate(self.coordinates):

                if idx == 0:
                    self.assertAlmostEqual(coordinate[1], self.coordinates[idx+2][1])

                if idx == 1:
                    self.assertAlmostEqual(coordinate[1], 0.0)

                if 14 >= idx > 3:
                    self.assertGreaterEqual(self.coordinates[idx][0], self.coordinates[idx-1][0])

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
