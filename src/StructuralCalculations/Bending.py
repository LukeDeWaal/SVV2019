import numpy as np
from src.Structure import CrossSection, get_crossectional_coordinates
"""
Bending Calculations
"""



class NumericalBending:

    def __init__(self, cross_section):

        self.__cross_section = cross_section

    def calculate_bending_stress(self, Mz, My):

        """:param Mz:  moment around z
           :param My: moment around y
           :param Izz: moment of inertia around z axis
           :param Iyy: moment of inertia around y axis

           :return: sigma_x: normal stress in x direction"""

        stresses = []

        Izz = self.__cross_section.real_MOI(axis='zz')
        Iyy = self.__cross_section.real_MOI(axis='yy')

        for boom in self.__cross_section:

            stiffener_locations = boom.get_position()
            sigma_x = Mz * stiffener_locations[1] / Izz + My * stiffener_locations[2] / Iyy
            stresses.append(sigma_x)

        return stresses

    def boom_areas_calculator(self, normal_stresses):

        ''':param normal_stresses: the combined normal stresses in given section at booms locations
           :param t: thickness of the skin of the aileron
           :param: stiffner_areas: original stiffner areas
           :param: stiffener_locations: stiffener coordinates
           :return boom area in given section'''

        stiffner_areas= np.array([boom.get_size() for boom in self.__cross_section])
        stiffner_locations=np.array([boom.get_position() for boom in self.__cross_section])
        t=self.__cross_section.get_skin_objects()[0].get_thickness()

        booms = np.array([])

        boom_stress_indexes = np.arange(17)
        back_boom_stress_indexes = boom_stress_indexes - 1
        front_boom_stress_indexes = boom_stress_indexes + 1

        # calculating the distances between booms
        a = stiffner_locations[2][front_boom_stress_indexes] - stiffner_locations[2][boom_stress_indexes]
        b = stiffner_locations[1][front_boom_stress_indexes] - stiffner_locations[1][boom_stress_indexes]
        distances = np.sqrt(a ** 2 + b ** 2)

        boom_areas = + t * distances[back_boom_stress_indexes] / 6 * (2 + normal_stresses[back_boom_stress_indexes] / normal_stresses[boom_stress_indexes]) \
                     + t * distances[boom_stress_indexes] / 6 * (2 + normal_stresses[front_boom_stress_indexes] / normal_stresses[boom_stress_indexes]) \
                     + stiffner_areas

        return boom_areas

    def calculate_bending_deflection(self):
        pass


if __name__ == "__main__":
    ha = 0.205
    Ca = 0.605
    h_stringer = 1.6 * 10 ** (-2)

    coordinates = get_crossectional_coordinates(Ca, ha, h_stringer)
    CS = CrossSection(coordinates, initial_areas=True)

    calculations = NumericalBending(CS)

    BendingStress = calculations.calculate_bending_stress()