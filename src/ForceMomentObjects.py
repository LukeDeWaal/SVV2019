"""
Create Force and moment objects here
"""
import numpy as np
import unittest


class Force(object):

    def __init__(self, forcevector: np.array = np.array([0,0,0]), positionvector: np.array = np.array([0,0,0])):
        """
        Object describing a force vector
        :param forcevector: Vector describing the magnitude and direction
        :param positionvector: Vector describing the position the force is applied to
        """

        self.__force = forcevector
        self.__pos = positionvector

    def get_position(self):
        return self.__pos

    def set_position(self, positionvector: np.array):
        self.__pos = positionvector

    def get_force(self):
        return self.__force

    def set_force(self, forcevector: np.array):
        self.__force = forcevector

    def get_direction(self):
        if self.get_magnitude() == 0.0:
            return self.get_force()
        return self.__force/np.linalg.norm(self.__force)

    def set_direction(self, direction: np.array):
        mag = self.get_magnitude()
        self.__force = direction*mag

    def get_magnitude(self):
        return np.linalg.norm(self.__force)

    def set_magnitude(self, magnitude):
        dirvec = self.get_direction()
        self.__force = magnitude*dirvec


class Moment(object):

    def __init__(self, momentvector: np.array = np.array([0,0,0]), positionvector: np.array = np.array([0,0,0])):
        """
        Object describing a moment / torque
        :param momentvector: Vector describing the direction and magnitude
        :param positionvector: Vector describing the position it is applied
        """

        self.__moment = momentvector
        self.__pos = positionvector

    def get_position(self):
        return self.__pos

    def set_position(self, positionvector: np.array):
        self.__pos = positionvector

    def get_moment(self):
        return self.__moment

    def set_moment(self, momentvector: np.array):
        self.__moment = momentvector

    def get_direction(self):
        if self.get_magnitude() == 0.0:
            return self.get_moment()
        return self.__moment / np.linalg.norm(self.__moment)

    def set_direction(self, direction: np.array):
        mag = self.get_magnitude()
        self.__moment = direction * mag

    def get_magnitude(self):
        return np.linalg.norm(self.__moment)

    def set_magnitude(self, magnitude):
        dirvec = self.get_direction()
        self.__moment = magnitude * dirvec



if __name__ == "__main__":

    class ForceMomentTestCases(unittest.TestCase):

        """
        Class Containing all Unit Tests
        """

        def setUp(self):
            self.F = Force()
            self.M = Moment()

        def test_position_getter_setter_methods(self):

            expected_positions = np.array([0, 0, 0])

            self.assertEqual(self.F.get_position().all(), expected_positions.all())
            self.assertEqual(self.M.get_position().all(), expected_positions.all())

            self.F.set_position(np.array([1,10,-3]))
            self.M.set_position(np.array([-73, 10, 2]))

            self.assertEqual(self.F.get_position().all(), np.array([1,10,-3]).all())
            self.assertEqual(self.M.get_position().all(), np.array([-73, 10, 2]).all())

        def test_vector_getter_setter_methods(self):

            expected_vectors = np.array([0, 0, 0])

            self.assertEqual(self.F.get_force().all(), expected_vectors.all())
            self.assertEqual(self.M.get_moment().all(), expected_vectors.all())

            self.F.set_force(np.array([10, -4, 3]))
            self.M.set_moment(np.array([-2, 8, 12]))

            self.assertEqual(self.F.get_force().all(), np.array([10, -4, 3]).all())
            self.assertEqual(self.M.get_moment().all(), np.array([-2, 8, 12]).all())

        def test_direction_getter_setter_methods(self):

            expected_dirs = np.array([0, 0, 0])

            self.assertEqual(self.F.get_direction().all(), expected_dirs.all())
            self.assertEqual(self.M.get_direction().all(), expected_dirs.all())

            self.F.set_force(np.array([-4, 3, -2]))
            self.M.set_moment(np.array([5, -2, 1]))

            F_dir = np.array([-4, 3, -2]) / np.linalg.norm(np.array([-4, 3, -2]))
            M_dir = np.array([5, -2, 1]) / np.linalg.norm(np.array([5, -2, 1]))

            self.assertAlmostEqual(self.F.get_direction().all(), F_dir.all())
            self.assertAlmostEqual(self.M.get_direction().all(), M_dir.all())

        def test_magnitude_getter_setter_methods(self):

            expected_magnitudes = 0

            self.assertEqual(self.F.get_magnitude(), expected_magnitudes)
            self.assertEqual(self.M.get_magnitude(), expected_magnitudes)

            self.F.set_force(np.array([10, -20, 50]))
            self.M.set_moment(np.array([5, 12, -20]))

            self.assertAlmostEqual(self.F.get_magnitude(), np.linalg.norm(np.array([10, -20, 50])))
            self.assertAlmostEqual(self.M.get_magnitude(), np.linalg.norm(np.array([5, 12, -20])))

            self.F.set_magnitude(100)
            self.M.set_magnitude(10)

            self.assertAlmostEqual(self.F.get_magnitude(), 100)
            self.assertAlmostEqual(self.M.get_magnitude(), 10)


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(ForceMomentTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
