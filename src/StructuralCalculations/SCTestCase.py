from src.StructuralCalculations.Bending import NumericalBending
from src.StructuralCalculations.Torsion import NumericalTorsion
from src.StructuralCalculations.Shear import NumericalShear
import unittest


class StructuralCalculationsTestCases(unittest.TestCase):

    def setUp(self):
        self.B = NumericalBending()
        self.T = NumericalTorsion()
        self.S = NumericalShear
