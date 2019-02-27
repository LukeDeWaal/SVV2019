import numpy as np
import unittest
from scipy.integrate import quad


def derive(f, h=10**(-5)):
    def fp(x):
        return (f(x+h)-f(x))/h

    return fp


def integrate(f):

    def integral(a, b, dx=10**(-5), steps=None):

        if steps is None and dx is not None:
            steps = (b-a)/dx

        elif dx is None and steps is not None:
            dx = (b-a)/steps

        else: raise ValueError

        I = 0

        for i in range(int(steps)):
            I += f(a+i*dx)*dx

        return I

    return integral


def factorial(n):
    if n == (1 or 0):
        return 1
    return factorial(n-1)*n


def pythagoras(a, b):
    return np.sqrt(a**2+b**2)


def step_function(x, x0):
    if x < x0:
        return 0.0
    else:
        return 1.0


def reLu(x, x0):
    if x >= x0:
        return x - x0
    else:
        return 0.0


def newtons_method(f, x0, maxiter=1000):
    fp = derive(f)
    x = x0
    i = 0
    while True:
        old_x = x
        x = x - f(x)/fp(x)

        if i >= maxiter:
            print("max Iterations Reached")
            break

        if np.abs(old_x - x) <= 10**(-7):
            break

        i += 1

    return x

#
# def coordinate_rotation(point: np.array, axis: np.array, angle: float):
#     from numpy import cos, sin
#     A = np.array([[axis[0]**2*(1-cos(angle)) + cos(angle), axis[1]*axis[0]*(1-cos(angle))-axis[2]*sin(angle), axis[2]*axis[0]*(1-cos(angle))+axis[1]*sin(angle)],
#                   [axis[1]*axis[0]*(1-cos(angle))+axis[2]*sin(angle), axis[1]**2*(1-cos(angle)) + cos(angle), axis[1]*axis[2]*(1-cos(angle))-axis[0]*sin(angle)],
#                   [axis[2]*axis[0]*(1-cos(angle))-axis[1]*sin(angle), axis[1]*axis[2]*(1-cos(angle))+axis[0]*sin(angle), axis[2]**2*(1-cos(angle)) + cos(angle)]])
#
#     return np.matmul(A, point.reshape((len(point), 1)))
#
#
# def coordinate_reflection(point: np.array, plane_normal: np.array):
#
#     a, b, c = plane_normal
#
#     A = np.array([[1-2*a**2, -2*a*b, -2*a*c],
#                   [-2*a*b, 1-2*b**2, -2*b*c],
#                   [-2*a*c, -2*b*c, 1-2*c**2]], dtype=float)
#
#     point = point.reshape((len(point), 1))
#
#     return np.matmul(A, point.astype(float))


def x_axis_rotation(point: np.array, angle: float) -> np.array:

    A = lambda t: np.array([[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -1.0*np.sin(t), np.cos(t)]], dtype=float)

    return np.matmul(A(angle), point.reshape((3,1)))


def xy_plane_reflection(point: np.array) -> np.array:

    A = np.array([[1,0,0],[0,1,0],[0,0,-1]], dtype=float)

    return np.matmul(A, point.reshape((3,1)))


def coordinate_transformation(point: np.array) -> np.array:

    p_i = x_axis_rotation(point, 28.0*np.pi/180.0)
    p_ii = xy_plane_reflection(p_i)

    return p_ii.reshape((1,3))


if __name__ == "__main__":

    class TestNumericalTools(unittest.TestCase):

        def setUp(self):

            self.f1 = lambda x: np.cos(x)
            self.f2 = lambda x: np.sin(x)
            self.f3 = lambda x: 3*x**2+4*x-4

            self.p1 = np.array([1,0,0])
            self.p2 = np.array([0,1,0])
            self.p3 = np.array([0,0,1])

        def test_derivatives(self):

            self.assertAlmostEqual(self.f1(1), derive(self.f2)(1), delta=10**(-3))
            self.assertAlmostEqual(self.f2(1), -1*derive(self.f1)(1), delta=10**(-3))

        def test_newtonsmethod(self):

            self.assertAlmostEqual(newtons_method(self.f3, -3.0), -2.0, delta=10**(-3))
            self.assertAlmostEqual(newtons_method(self.f3, 2.0), 2.0/3.0, delta=10**(-3))

        def test_coordinate_transformations(self):

            rot_p1 = x_axis_rotation(self.p1, np.pi)
            self.assertEqual(self.p1.all(), rot_p1.all())

            ref_p1 = xy_plane_reflection(self.p1)
            self.assertEqual(-1.0*self.p1.all(), ref_p1.all())

            rot_p2 = x_axis_rotation(self.p2, np.pi)
            self.assertEqual(self.p2.all(), rot_p2.all())

            ref_p2 = xy_plane_reflection(self.p2)
            self.assertEqual(-1.0*self.p2.all(), ref_p2.all())

            rot_p3 = x_axis_rotation(self.p3, np.pi)
            self.assertEqual(self.p3.all(), rot_p3.all())

            ref_p3 = xy_plane_reflection(self.p3)
            self.assertEqual(-1.0*self.p3.all(), ref_p3.all())

            tot_1 = coordinate_transformation(self.p1)
            self.assertEqual(self.p1.all(), tot_1.all())

            tot_2 = coordinate_transformation(self.p2)
            self.assertAlmostEqual(np.array([0.0, 0.88295, 0.4695]).all(), tot_2.all())

            tot_3 = coordinate_transformation(self.p3)
            self.assertAlmostEqual(np.array([0.0, 0.4695, -0.88295]).all(), tot_3.all())


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNumericalTools)
        unittest.TextTestRunner(verbosity=2).run(suite)


    run_TestCases()
