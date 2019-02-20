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

    def get_position(self) -> np.array:
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


class DistributedLoad(object):

    def __init__(self, magnitude: float, startpos: np.array, endpos: np.array, direction: np.array, N):

        self.__magnitude = magnitude
        self.__direction = np.array(direction) if type(direction) != np.array else direction
        self.__start = np.array(startpos) if type(startpos) != np.array else startpos
        self.__end = np.array(endpos) if type(endpos) != np.array else endpos
        self.__N = N

        self.__discretized_forces = self.discretize()

    def discretize(self):
        forces = []
        x_positions = np.linspace(self.__start[0], self.__end[0], self.__N)
        y_positions = np.linspace(self.__start[1], self.__end[1], self.__N)
        z_positions = np.linspace(self.__start[2], self.__end[2], self.__N)
        for i in range(self.__N):
            position = np.array([x_positions[i], y_positions[i], z_positions[i]])
            force_magnitude = self.__magnitude*self.get_length()/self.get_N()
            forces.append(Force(force_magnitude*self.__direction/np.linalg.norm(self.__direction), position))

        return forces

    def get_discretized_forces(self):
        return self.__discretized_forces

    def get_magnitude(self):
        return self.__magnitude

    def get_direction(self):
        return self.__direction

    def get_length(self):
        return np.linalg.norm(self.__start - self.__end)

    def get_N(self):
        return self.__N


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
            self.q = DistributedLoad(100, [0,0,0], [0,0,10], [0, -1, 0], 5)

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

        def test_distributed_load_discretization(self):

            discr = self.q.get_discretized_forces()
            for force in discr:
                self.assertAlmostEqual(force.get_magnitude(), self.q.get_magnitude()*self.q.get_length()/self.q.get_N())


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(ForceMomentTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
