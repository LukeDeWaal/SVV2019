"""
Idealization Objects (skin, boom)
"""

import numpy as np
import unittest


class Boom(object):

    def __init__(self, size: float = 1.0, density: float = 0.0, position: np.array = np.array([0,0,0]), which='Stiffener', label=0):
        """
        Create a boom object
        :param mass: Mass of boom
        :param size: Size of boom
        :param position: Position Vector of boom
        """

        self.__density = density
        self.__size = size
        self.__pos = position
        self.__type = which
        self.__label = label

    def __str__(self):
        return f"Type: {self.get_type()}, Mass: {self.get_mass()}, Size: {self.get_size()} \nPosition: {self.get_position()}"

    def set_type(self, boom_type):
        self.__type = boom_type

    def get_type(self):
        return self.__type

    def get_label(self):
        return self.__label

    def set_label(self, label):
        self.__label = label

    def get_mass(self):
        return self.__density * self.__size

    def get_density(self):
        return self.__density

    def set_density(self, density):
        self.__density = density

    def get_size(self):
        return self.__size

    def set_size(self, size):
        self.__size = size

    def get_position(self):
        return self.__pos

    def set_position(self, position):
        self.__pos = position

    def area_MOI(self, axis1, axis2=None):
        pass
    
    def det_distance(self, boom2):
        """Determines the vector going from boom 2 TOO SELF in x,y and z directions
        and returns it as a vector"""
        return (self.get_position()- boom2.get_position())

class StraightSkin(object):

    def __init__(self, thickness: float = 0.001, startpos: np.array = np.array([0, 0, 0]), endpos: np.array = np.array([0, 0, 0]), density: float = 0.0):
        """
        Create a skin object (to connect booms)
        :param mass: Mass of skin
        :param thickness: Thickness of skin
        :param startpos: Startposition of skin
        :param endpos: Endposition of skin

        Skin is a straight line running from start untill end

        """

        self.__density = density
        self.__t = thickness
        self.__start = startpos
        self.__end = endpos

    def get_density(self):
        return self.__density

    def set_density(self, density):
        self.__density = density

    def get_mass(self):
        return self.__t*np.linalg.norm(self.__start - self.__end)*self.__density

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


if __name__ == "__main__":

    class IdealizationTestCases(unittest.TestCase):

        """
        Class Containing all Unit Tests
        """

        def setUp(self):

            self.boom = Boom()
            self.s_skin = StraightSkin()

            self.longMessage = True

        def test_mass_and_density_setter_getter_methods(self):

            #Initialized Masses should be 0
            expected_masses = 0

            boom_mass = self.boom.get_mass()
            skin_mass = self.s_skin.get_mass()

            self.assertEqual(expected_masses, boom_mass)
            self.assertEqual(expected_masses, skin_mass)

            #Set new densities
            new_densities = [2800.0, 2720.0]

            self.boom.set_density(new_densities[0])
            self.s_skin.set_density(new_densities[1])

            boom_dens = self.boom.get_density()
            s_skin_dens = self.s_skin.get_density()

            self.assertEqual(new_densities[0], boom_dens)
            self.assertEqual(new_densities[1], s_skin_dens)

            #Get masses

            self.s_skin.set_position([10, 0, 0], 'end')

            boom_mass = self.boom.get_mass()
            skin_mass = self.s_skin.get_mass()

            self.assertAlmostEqual(boom_mass, self.boom.get_size()*self.boom.get_density())
            self.assertAlmostEqual(skin_mass, self.s_skin.get_thickness()*self.s_skin.get_length()*self.s_skin.get_density())



        def test_position_setter_getter_methods(self):

            #Initialized positions should be [0,0,0]
            expected_position = np.array([0, 0, 0])

            boom_pos = self.boom.get_position()
            s_skin_pos = self.s_skin.get_position(which='start')

            self.assertEqual(expected_position.all(), boom_pos.all())
            self.assertEqual(expected_position.all(), s_skin_pos.all())

            #Set new start positions
            new_positions = [np.array([10, 3, 23]), np.array([4, 10, -4]), np.array([28, -1, -1000])]

            self.boom.set_position(new_positions[0])
            self.s_skin.set_position(new_positions[1], which='start')

            boom_pos = self.boom.get_position()
            s_skin_pos = self.s_skin.get_position(which='start')

            self.assertEqual(new_positions[0].all(), boom_pos.all())
            self.assertEqual(new_positions[1].all(), s_skin_pos.all())

            # Set new end positions
            new_positions = [np.array([44, -13, 54]), np.array([-12, -31, 26])]

            self.s_skin.set_position(new_positions[0], which='end')

            s_skin_pos = self.s_skin.get_position(which='end')

            self.assertEqual(new_positions[0].all(), s_skin_pos.all())

        def test_size_setter_getter_methods(self):

            #Initialized thicknesses

            boom_size = self.boom.get_size()
            s_skin_thickness = self.s_skin.get_thickness()

            self.assertEqual(1.0, boom_size)
            self.assertEqual(0.001, s_skin_thickness)

            #Set new masses
            new_thicknesses = [110, -3, 233]

            self.boom.set_size(new_thicknesses[0])
            self.s_skin.set_thickness(new_thicknesses[1])

            boom_size = self.boom.get_size()
            s_skin_thickness = self.s_skin.get_thickness()

            self.assertEqual(new_thicknesses[0], boom_size)
            self.assertEqual(new_thicknesses[1], s_skin_thickness)

        def test_length_setter_getter_methods(self):

            #Initialized Values should be 0 and 0
            expected_length = 0
            expected_radius = 0

            straight_len = self.s_skin.get_length()

            self.assertEqual(expected_length, straight_len)

            #Setting New Lengths and Radius

            self.s_skin.set_position(np.array([1,1,1]), which='end')

            new_l = self.s_skin.get_length()

            self.assertAlmostEqual(new_l, np.linalg.norm([1,1,1]))


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(IdealizationTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
