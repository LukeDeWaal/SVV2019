"""
Build the aileron structural idealization here (using skin, booms, etc)
"""

from src.Idealizations import *
import matplotlib.pyplot as plt



class AssembledStructure(object):

    def __init__(self, *objects: "Booms and/or Skin"):

        self.__objects = objects
        self.__structure = None

