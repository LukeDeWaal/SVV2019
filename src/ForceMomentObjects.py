"""
Create Force and moment objects here
"""
import numpy as np


class Force(object):

    def __init__(self, forcevector: np.array, positionvector: np.array):
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

    def __init__(self, momentvector: np.array, positionvector: np.array):
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
        return self.__moment / np.linalg.norm(self.__moment)

    def set_direction(self, direction: np.array):
        mag = self.get_magnitude()
        self.__moment = direction * mag

    def get_magnitude(self):
        return np.linalg.norm(self.__moment)

    def set_magnitude(self, magnitude):
        dirvec = self.get_direction()
        self.__moment = magnitude * dirvec
