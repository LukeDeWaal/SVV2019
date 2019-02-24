import numpy as np
import unittest


def derive(f, h=10**(-5)):
    def fp(x):
        return (f(x+h)-f(x))/h

    return fp


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


def integrate(q):

    def F(z, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = len(z)-1
        s = 0.0
        dz = z[1] - z[0]
        for idx in range(len(z[start_idx:end_idx])):
            s += (q(z)[idx]/1.0)*dz
        s = 1.0*s
        return s

    return F


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


if __name__ == "__main__":

    class TestNumericalTools(unittest.TestCase):

        def setUp(self):

            self.f1 = lambda x: np.cos(x)
            self.f2 = lambda x: np.sin(x)
            self.f3 = lambda x: 3*x**2+4*x-4

        def test_derivatives(self):

            self.assertAlmostEqual(self.f1(1), derive(self.f2)(1), delta=10**(-3))
            self.assertAlmostEqual(self.f2(1), -1*derive(self.f1)(1), delta=10**(-3))

        def test_newtonsmethod(self):

            self.assertAlmostEqual(newtons_method(self.f3, -3.0), -2.0, delta=10**(-3))
            self.assertAlmostEqual(newtons_method(self.f3, 2.0), 2.0/3.0, delta=10**(-3))


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNumericalTools)
        unittest.TextTestRunner(verbosity=2).run(suite)


    run_TestCases()
