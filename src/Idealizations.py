"""
Idealization Objects (skin, boom)
"""

import numpy as np
import unittest


class Boom(object):

    def __init__(self, mass: float = 0, size: float = 0, position: np.array = np.array([0,0,0])):
        """
        Create a boom object
        :param mass: Mass of boom
        :param size: Size of boom
        :param position: Position Vector of boom
        """

        self.__mass = mass
        self.__size = size
        self.__pos = position

    def get_mass(self):
        return self.__mass

    def set_mass(self, mass):
        self.__mass = mass

    def get_size(self):
        return self.__size

    def set_size(self, size):
        self.__size = size

    def get_position(self):
        return self.__pos

    def set_position(self, position):
        self.__pos = position


class StraightSkin(object):

    def __init__(self, mass: float = 0, thickness: float = 0, startpos: np.array = np.array([0, 0, 0]), endpos: np.array = np.array([0, 0, 0])):
        """
        Create a skin object (to connect booms)
        :param mass: Mass of skin
        :param thickness: Thickness of skin
        :param startpos: Startposition of skin
        :param endpos: Endposition of skin

        Skin is a straight line running from start untill end

        """

        self.__mass = mass
        self.__t = thickness
        self.__start = startpos
        self.__end = endpos

    def get_mass(self):
        return self.__mass

    def set_mass(self, mass):
        self.__mass = mass

    def get_thickness(self):
        return self.__t

    def set_thickness(self, t):
        self.__t = t

    def get_length(self):
        return np.linalg.norm(self.__end - self.__start)

    def set_position(self, position, which='start'):
        if which == 'start':
            self.__start = position

        elif which == 'end':
            self.__end = position

    def get_position(self, which='start'):
        if which == 'start':
            return self.__start
        elif which == 'end':
            return self.__end


class CurvedSkin(StraightSkin):
    #
    def __init__(self, mass: float = 0, thickness: float = 0, startpos: np.array = np.array([0, 0, 0]), endpos: np.array = np.array([0, 0, 0]), radius: float = 0):
        """
        Create a curved skin object (to connect booms)
        :param mass: Mass of skin
        :param thickness: Thickness of skin
        :param radius: Radius of curvature
        :param angle: Angle over which curvature acts
        """

        StraightSkin.__init__(self, mass, thickness, startpos, endpos)
        self.__r = radius

    def get_radius(self):
        return self.__r

    def set_radius(self, radius):
        self.__r = radius


if __name__ == "__main__":

    class IdealizationTestCases(unittest.TestCase):

        def setUp(self):

            self.boom = Boom()
            self.s_skin = StraightSkin()
            self.c_skin = CurvedSkin()

            self.longMessage = True

        def test_mass_setter_getter_methods(self):

            #Initialized Masses should be 0
            expected_masses = 0

            boom_mass = self.boom.get_mass()
            s_skin_mass = self.s_skin.get_mass()
            c_skin_mass = self.c_skin.get_mass()

            self.assertEqual(expected_masses, boom_mass)
            self.assertEqual(expected_masses, s_skin_mass)
            self.assertEqual(expected_masses, c_skin_mass)

            #Set new masses
            new_masses = [10, 3, 23]

            self.boom.set_mass(new_masses[0])
            self.s_skin.set_mass(new_masses[1])
            self.c_skin.set_mass(new_masses[2])

            boom_mass = self.boom.get_mass()
            s_skin_mass = self.s_skin.get_mass()
            c_skin_mass = self.c_skin.get_mass()

            self.assertEqual(new_masses[0], boom_mass)
            self.assertEqual(new_masses[1], s_skin_mass)
            self.assertEqual(new_masses[2], c_skin_mass)

        def test_position_setter_getter_methods(self):

            #Initialized positions should be [0,0,0]
            expected_position = np.array([0, 0, 0])

            boom_pos = self.boom.get_position()
            s_skin_pos = self.s_skin.get_position(which='start')
            c_skin_pos = self.c_skin.get_position(which='start')

            self.assertEqual(expected_position.all(), boom_pos.all())
            self.assertEqual(expected_position.all(), s_skin_pos.all())
            self.assertEqual(expected_position.all(), c_skin_pos.all())

            #Set new positions
            new_positions = [np.array([10, 3, 23]), np.array([4, 10, -4]), np.array([28, -1, -1000])]

            self.boom.set_position(new_positions[0])
            self.s_skin.set_position(new_positions[1], which='start')
            self.c_skin.set_position(new_positions[2], which='start')

            boom_pos = self.boom.get_position()
            s_skin_pos = self.s_skin.get_position(which='start')
            c_skin_pos = self.c_skin.get_position(which='start')

            self.assertEqual(new_positions[0].all(), boom_pos.all())
            self.assertEqual(new_positions[1].all(), s_skin_pos.all())
            self.assertEqual(new_positions[2].all(), c_skin_pos.all())

        def test_size_setter_getter_methods(self):

            #Initialized thicknesses should be 0
            expected_thicknesses = 0

            boom_size = self.boom.get_size()
            s_skin_thickness = self.s_skin.get_thickness()
            c_skin_thickness = self.c_skin.get_thickness()

            self.assertEqual(expected_thicknesses, boom_size)
            self.assertEqual(expected_thicknesses, s_skin_thickness)
            self.assertEqual(expected_thicknesses, c_skin_thickness)

            #Set new masses
            new_thicknesses = [110, -3, 233]

            self.boom.set_size(new_thicknesses[0])
            self.s_skin.set_thickness(new_thicknesses[1])
            self.c_skin.set_thickness(new_thicknesses[2])

            boom_size = self.boom.get_size()
            s_skin_thickness = self.s_skin.get_thickness()
            c_skin_thickness = self.c_skin.get_thickness()
<<<<<<< HEAD

            self.assertEqual(new_thicknesses[0], boom_size)
            self.assertEqual(new_thicknesses[1], s_skin_thickness)
            self.assertEqual(new_thicknesses[2], c_skin_thickness)

        def test_length_setter_getter_methods(self):

            #Initialized Values should be 0 and 0
            expected_length = 0
            expected_radius = 0

            straight_len = self.s_skin.get_length()
            curved_radius = self.c_skin.get_radius()

            self.assertEqual(expected_length, straight_len)
            self.assertEqual(expected_radius, curved_radius)

            #Setting New Lengths and Radius
            self.c_skin.set_radius(5)
            self.s_skin.set_position(np.array([1,1,1]), which='end')

            new_r = self.c_skin.get_radius()
            new_l = self.s_skin.get_length()

            self.assertEqual(new_r, 5)
            self.assertAlmostEqual(new_l, np.linalg.norm([1,1,1]), places=5)
=======

            self.assertEqual(new_thicknesses[0], boom_size)
            self.assertEqual(new_thicknesses[1], s_skin_thickness)
            self.assertEqual(new_thicknesses[2], c_skin_thickness)
>>>>>>> 9887a445fc1e7de04c2498f0cc91968389768e00

    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(IdealizationTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
