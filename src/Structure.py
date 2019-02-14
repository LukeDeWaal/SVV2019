"""
Build the aileron structural idealization here (using skin, booms, etc)
"""

from src.Idealizations import *
import matplotlib.pyplot as plt
import unittest
import numpy as np


class AssembledStructure(object):

    def __init__(self, *objects: "Booms and/or Skin"):

        self.__objects = objects
        self.__structure = None


if __name__ == "__main__":

    class StructureTestCases(unittest.TestCase):

        def setUp(self):

            self.structure = AssembledStructure()


    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(IdealizationTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
