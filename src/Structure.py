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

    def __init__(self, boom_coordinates, sparcap_coordinates=np.array([[ha/2, ha/2],[-ha/2, ha/2]]), x_coordinate=0):

        # Pointer
        self.__cur = 0

        # X-Coordinate
        self.__x = x_coordinate
        self.__x_vector = np.ones((len(boom_coordinates[:, 0]), 1)) * self.__x

        # Boom Coordinates
        self.__stiffener_coordinates = np.concatenate((self.__x_vector, boom_coordinates), axis=1)
        self.__sparcap_coordinates   = np.concatenate((np.ones((2, 1)) * self.__x, sparcap_coordinates), axis=1) if sparcap_coordinates is not False else None

        # Creating all Boom Objects
        self.__stiffener_booms = self.__initialize_boom_objects(self.__stiffener_coordinates)
        self.__spar_booms = self.__initialize_spar_caps(self.__sparcap_coordinates) if self.__sparcap_coordinates is not None else None
        self.__all_booms = self.__stiffener_booms

        if self.__spar_booms is not None:
            self.__all_booms.insert(0, self.__spar_booms[0])
            self.__all_booms.insert(4, self.__spar_booms[1])

        for idx in range(len(self.__all_booms)):
            self.__all_booms[idx].set_label(idx)

        # Creating all Skin Objects between the Boom Objects
        self.__skin_objects = self.__initialize_skin_objects(self.__all_booms)

        # Calculate initial centroid
        self.__centroid = self.get_centroid()

        # Calculate amount of booms
        self.__N_booms = len(self.get_stiffener_objects())
        self.__N_spars = len(self.get_spar_caps()) if self.__spar_booms is not None else 0
        self.__N_total = self.__N_booms + self.__N_spars


    @staticmethod
    def __initialize_boom_objects(coordinates) -> list:
        return [Boom(density=1, size=1, position=coordinate, which='Stiffener') for coordinate in coordinates]

    @staticmethod
    def __initialize_spar_caps(coordinates) -> list:
        return [Boom(density=1, size=2, position=coordinate, which='Sparcap') for coordinate in coordinates]

    @staticmethod
    def __initialize_skin_objects(boom_objects) -> list:
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

    def __str__(self) -> str:
        return f"# Booms: {self.get_N_booms()}, # Spar Caps: {self.get_N_spars()} \n"

    def __len__(self) -> int:
        return self.get_N_objects()

    def __getitem__(self, index):
        return self.__all_booms[index]

    def __setitem__(self, index, value):
        if type(value) == Boom:
            self.__all_booms[index] = value
        else:
            raise TypeError

    def __iter__(self):
        self.__cur = 0
        return self

    def __next__(self):
        if self.__cur >= self.get_N_objects():
            raise StopIteration

        result = self.get_all_booms()[self.__cur]
        self.__cur += 1
        return result

    def get_N_booms(self) -> int:
        return self.__N_booms

    def get_N_spars(self) -> int:
        return self.__N_spars

    def get_N_objects(self) -> int:
        return self.get_N_booms() + self.get_N_spars()

    def set_x(self, x):
        """
        Set the x-coordinate of the crosssection
        :param x: X Coordinate
        :return: None
        """
        for boom in self.get_all_booms():
            old_position = boom.get_position()
            old_position[0] = x
            boom.set_position(old_position)

    def get_x(self) -> float or int:
        """
        Check the x-value of the crosssection
        :return: x coordinate
        """
        return self.get_stiffener_objects()[0].get_position()[0]

    def get_centroid(self) -> float or int:
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

    def area_MOI(self, axes: str) -> float or int:
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

    def update_boom_area(self, idx: int, size_increment: int or float):
        """
        :param idx: Index of boom item (top sparcap is idx=0, then moves counterclockwise)
        :return: None
        """
        old_size = self.get_all_booms()[idx].get_size()
        self.get_all_booms()[idx].set_size(old_size + size_increment)

    def get_mass(self) -> float or int:
        """
        Get total summed up mass of booms and skins
        :return: Total Mass
        """
        return sum([skin.get_mass() for skin in self.get_skin_objects()]) + \
               sum([boom.get_mass() for boom in self.get_all_booms()])

    def get_all_booms(self) -> list:
        """
        :return: Stiffener + Spar Cap Objects
        """
        return self.__all_booms

    def get_spar_caps(self) -> list:
        """
        :return: Spar Cap Objects
        """
        return [self.get_all_booms()[0], self.get_all_booms()[4]]

    def get_stiffener_objects(self) -> list:
        """
        :return: Stiffener Objects
        """
        return self.get_all_booms()[1:4] + self.get_all_booms()[5:]

    def get_skin_objects(self) -> list:
        """
        :return: Skin Objects
        """
        return self.__skin_objects

    def get_all_coordinates(self) -> np.array:
        """
        :return: Array of crosssectional coordinates
        """
        return np.array([boom.get_position() for boom in self.get_all_booms()])

    def get_stiffener_coordinates(self):
        """
        :return: Array of stiffener coordinates
        """
        return self.__stiffener_coordinates

    def get_sparcap_coordinates(self):
        """
        :return: Array of sparcap coordinates
        """
        return self.__sparcap_coordinates

    def calculate_shear_centre(self):
        pass


class FullModel(object):

    def __init__(self, coordinates, xrange, N, sparcaps: bool or np.array = np.array([[ha/2, ha/2],[-ha/2, ha/2]])):

        self.__sections = [None]*N
        self.__cur = 0
        self.__boomcoordinates = coordinates
        self.__xrange = xrange
        self.__N = N

        self.__assemble_structure(sparcaps)

    def __len__(self):
        return self.__N

    def __str__(self):
        string = ""
        for section in self.get_sections():
            string += str(section)

    def __iter__(self):
        self.__cur = 0
        return self

    def __next__(self):

        if self.__cur >= self.get_N_sections():
            raise StopIteration()

        else:
            result = self.get_sections()[self.__cur]
            self.__cur += 1
            return result

    def get_N_sections(self):
        return self.__N

    def __assemble_structure(self, sparcaps):

        for idx, xi in enumerate(np.linspace(self.__xrange[0], self.__xrange[1], self.__N)):
            self.__sections[idx] = CrossSection(self.__boomcoordinates, sparcap_coordinates=sparcaps, x_coordinate=xi)

    def get_all_boom_coordinates(self):

        coordinates = tuple([section.get_all_coordinates() for section in self.get_sections()])
        return np.concatenate(coordinates, axis=0)

    def calculate_reaction_forces(self):
        pass

    @staticmethod
    def __parametrize_line(v1: np.array, v2: np.array):
        def line(t):
            return v1 + (v2-v1)*t
        return line

    def plot_structure(self, ax):

        # Boom Coordinates
        coordinates = self.get_all_boom_coordinates()
        xboomplot = coordinates[:, 0]
        yboomplot = coordinates[:, 1]
        zboomplot = coordinates[:, 2]

        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-0.1, 0.6)
        ax.set_zlim3d(-0.3, 0.3)

        # Lines around crosssection
        xlineplot_1 = [[] for _ in range(self.__N)]
        ylineplot_1 = [[] for _ in range(self.__N)]
        zlineplot_1 = [[] for _ in range(self.__N)]

        for idx, section in enumerate(self.get_sections()):
            section_coordinates = section.get_all_booms()
            for boom in section_coordinates:
                position = boom.get_position()
                xlineplot_1[idx].append(position[0])
                ylineplot_1[idx].append(position[1])
                zlineplot_1[idx].append(position[2])
            xlineplot_1[idx].append(section_coordinates[0].get_position()[0])
            ylineplot_1[idx].append(section_coordinates[0].get_position()[1])
            zlineplot_1[idx].append(section_coordinates[0].get_position()[2])

        # Lines through crosssection
        xlineplot_2 = [[] for _ in range(17)]
        ylineplot_2 = [[] for _ in range(17)]
        zlineplot_2 = [[] for _ in range(17)]

        for section in self.get_sections():
            section_coordinates = section.get_all_booms()
            for idx, boom in enumerate(section_coordinates):
                position = boom.get_position()
                xlineplot_2[idx].append(position[0])
                ylineplot_2[idx].append(position[1])
                zlineplot_2[idx].append(position[2])

        ax.scatter3D(xboomplot, zboomplot, yboomplot, s=40, c='k')

        for i in range(self.__N):
            ax.plot(xlineplot_1[i], zlineplot_1[i], ylineplot_1[i], 'k')

        for i in range(17):
            ax.plot(xlineplot_2[i], zlineplot_2[i], ylineplot_2[i], 'k')

    def get_mass(self):
        return sum([section.get_mass() for section in self.get_sections()])

    def get_sections(self):
        return self.__sections


if __name__ == "__main__":

    coordinates = get_crossectional_coordinates(Ca, ha, h_stringer)
    crosssection = CrossSection(coordinates, )

    class StructureTestCases(unittest.TestCase):

        def setUp(self):

            self.ha = 0.205
            self.Ca = 0.605
            self.h_stringer = 1.6 * 10 ** (-2)

            self.coordinates = get_crossectional_coordinates(self.Ca, self.ha, self.h_stringer)

            self.crosssection = CrossSection(self.coordinates)
            self.model = FullModel(self.coordinates, (0, 2.661), 100)

            self.sqcoordinates = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
            self.sqmodel = FullModel(self.sqcoordinates, (-0.5, 0.5), 3, sparcaps=False)

            self.beamcoordinates = np.array([[-2, -1],[2, -1], [2, 1], [-2, 1]])
            self.beammodel = FullModel(self.beamcoordinates, (-1, 1), 3, sparcaps=False)

            self.logcoordinates = np.concatenate((np.log((1 + np.random.randint(np.pi, 2 * np.pi) * np.arange(0, 1, 0.1)).reshape(10, 1)), np.arange(0, 1, 0.1).reshape(10, 1)), axis=1)
            self.logmodel = FullModel(self.logcoordinates, (-1, 1), 5, sparcaps=False)


        def test_plots(self):

            for model in [self.model, self.sqmodel, self.beammodel, self.logmodel]:
                fig = plt.figure()
                ax = Axes3D(fig)
                model.plot_structure(ax)

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

            for boom in self.crosssection.get_stiffener_objects():
                self.assertEqual(boom.get_position()[0], 4.0)

        def test_centroid_calculaton(self):

            centroid = self.crosssection.get_centroid()

            self.assertAlmostEqual(centroid[0], self.crosssection.get_x())
            self.assertAlmostEqual(centroid[1], 0.0)
            self.assertLessEqual(centroid[2], self.Ca/2)

        def test_MOI(self):
            axes = ['z', 'y', 'zy']

            for idx, axis in enumerate(axes):
                # Cube Model
                for section in self.sqmodel:
                    MOI = section.area_MOI(axes=axis)
                    if axis == ('z' or 'y'):
                        self.assertAlmostEqual(MOI, 1.0)
                    elif axis == 'zy':
                        self.assertAlmostEqual(MOI, 0.0)

                # Beam Model
                for section in self.beammodel:
                    MOI = section.area_MOI(axes=axis)
                    if axis == 'z':
                        self.assertAlmostEqual(MOI, 16.0)
                    elif axis == 'y':
                        self.assertAlmostEqual(MOI, 4.0)
                    elif axis == 'zy':
                        self.assertAlmostEqual(MOI, 0.0)

                # Log Model
                for section in self.logmodel:
                    MOI = section.area_MOI(axes=axis)

                    if axis == 'zy':
                        self.assertNotEqual(MOI, 0.0)

        def test_boom_update(self):

            for idx in range(len(self.crosssection)):
                if self.crosssection[idx].get_type() == "Stiffener":
                    self.assertEqual(self.crosssection[idx].get_size(), 1.0)
                elif self.crosssection[idx].get_type() == "Sparcap":
                    self.assertEqual(self.crosssection[idx].get_size(), 2.0)
                else:
                    raise TypeError

                self.crosssection.update_boom_area(idx, 1.5)

            for boom in self.crosssection:
                if boom.get_type() == 'Stiffener':
                    self.assertEqual(boom.get_size(), 2.5)

                elif boom.get_type() == "Sparcap":
                    self.assertEqual(boom.get_size(), 3.5)

                else:
                    raise TypeError

        def test_get_coordinates(self):

            cs_coordinates = self.crosssection.get_stiffener_coordinates()

            for i in range(len(self.coordinates)):
                for j in range(len(self.coordinates[i])):
                    self.assertAlmostEqual(self.coordinates[i][j], cs_coordinates[i][j+1])

    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(StructureTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
