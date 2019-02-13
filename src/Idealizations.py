"""
Idealization Objects (skin, boom)
"""

import numpy as np


class Boom(object):

    def __init__(self, mass: float, size: float, position: np.array):
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

    def __init__(self, mass: float, thickness: float, startpos: np.array, endpos: np.array):
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


class CurvedSkin(object):

    def __init__(self, mass: float, thickness: float, radius: float, angle: float, startpos: np.array):
        """
        Create a curved skin object (to connect booms)
        :param mass: Mass of skin
        :param thickness: Thickness of skin
        :param radius: Radius of curvature
        :param angle: Angle over which curvature acts
        """

        self.__mass = mass
        self.__t = thickness
        self.__r = radius
        self.__theta = angle
        self.__pos = startpos

    def get_mass(self):
        return self.__mass

    def set_mass(self, mass):
        self.__mass = mass

    def get_thickness(self):
        return self.__t

    def set_thickness(self, t):
        self.__t = t

    def get_angle(self):
        return self.__theta

    def set_angle(self, angle):
        self.__theta = angle

    def get_radius(self):
        return self.__r

    def set_radius(self, radius):
        self.__r = radius

    def get_position(self):
        return self.__pos

    def set_position(self, position):
        self.__pos = position
