"""
Build the aileron structural idealization here (using skin, booms, etc)
"""

from src.Idealizations import *
import matplotlib.pyplot as plt
import unittest
import numpy as np


class CrossSection(object):

    def __init__(self, *objects: "Booms and/or Skin"):

        pass

    def area_MOI(self, axis1, axis2=None):
        pass

    def get_mass(self):
        pass

    def get_objects(self):
        pass

    def calculate_shear_centre(self):
        pass


class FullModel(object):

    def __init__(self, *crosssections, **forcesandmoments):
        pass

    def assemble_structure(self):
        pass

    def calculate_reaction_forces(self):
        pass

    def plot_structure(self):
        pass

    def get_mass(self):
        pass

    def get_objects(self):
        pass

if __name__ == "__main__":

    class StructureTestCases(unittest.TestCase):

        def setUp(self):

            self.crosssection = CrossSection()


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(IdealizationTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
