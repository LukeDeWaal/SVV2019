import numpy as np
from src.Structure import CrossSection, get_crossectional_coordinates, FullModel
from src.AppliedForcesMoments import moment_functions, distance_dict, force_dict
import unittest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
Bending Calculations
"""


class NumericalBending:

    def __init__(self, cross_section):

        self.__cross_section = cross_section

    def calculate_bending_stress(self, Mz, My):

        """
        :param Mz:  moment around z
        :param My: moment around y
        :param Izz: moment of inertia around z axis
        :param Iyy: moment of inertia around y axis

        :return: sigma_x: normal stress in x direction
        """

        stresses = []

        Izz = self.__cross_section.real_MOI(axes='zz')
        Iyy = self.__cross_section.real_MOI(axes='yy')

        centroid = self.__cross_section.get_centroid()

        for boom in self.__cross_section:

            stiffener_locations = boom.get_position()
            x = stiffener_locations[0]
            sigma_x = Mz(x) * (centroid[1] - stiffener_locations[1]) / Izz + My(x) * (centroid[2] - stiffener_locations[2]) / Iyy
            stresses.append(sigma_x)

        return np.array(stresses)

    def boom_areas_calculator(self, normal_stresses):

        ''':param normal_stresses: the combined normal stresses in given section at booms locations
           :param t: thickness of the skin of the aileron
           :param: stiffner_areas: original stiffner areas
           :param: stiffener_locations: stiffener coordinates
           :return boom area in given section'''

        stiffner_areas= np.array([boom.get_size() for boom in self.__cross_section])
        stiffner_locations=np.array([boom.get_position() for boom in self.__cross_section])
        t = self.__cross_section.get_skin_objects()[0].get_thickness()


        boom_stress_indexes = np.arange(17, dtype=int)
        back_boom_stress_indexes = np.arange(-1, 16, dtype=int)
        front_boom_stress_indexes = np.array([i for i in range(1, 17)]+[0], dtype=int)

        # calculating the distances between booms
        a = stiffner_locations[:, 2][front_boom_stress_indexes] - stiffner_locations[:, 2][boom_stress_indexes]
        b = stiffner_locations[:, 1][front_boom_stress_indexes] - stiffner_locations[:, 1][boom_stress_indexes]
        distances = np.sqrt(a ** 2 + b ** 2)

        #print(normal_stresses[boom_stress_indexes])

        boom_areas = + t * distances[back_boom_stress_indexes] / 6 * (2 + normal_stresses[back_boom_stress_indexes] / normal_stresses[boom_stress_indexes]) \
                     + t * distances[boom_stress_indexes] / 6 * (2 + normal_stresses[front_boom_stress_indexes] / normal_stresses[boom_stress_indexes]) \
                     + stiffner_areas

        for idx, boom in enumerate(self.__cross_section):
            boom.set_size(boom_areas[idx])

            # if boom.get_size() < 0:
            #     print(boom)

        return boom_areas


if __name__ == "__main__":

    ha = 0.205
    Ca = 0.605
    h_stringer = 1.6 * 10 ** (-2)

    coordinates = get_crossectional_coordinates(Ca, ha, h_stringer)
    CS = CrossSection(coordinates, initial_areas=True, x_coordinate=1.0, transform=True)

    Model = FullModel(coordinates, (0.0001, 2.661), 10)

    mx, my, mz = moment_functions(x_vals=distance_dict, forces=force_dict)

    areas = []

    for section in Model:
        calculations = NumericalBending(section)

        BendingStress = calculations.calculate_bending_stress(My=my, Mz=mz)
        Areas = calculations.boom_areas_calculator(BendingStress)
        areas.append(Areas)

    class BendingTestCases(unittest.TestCase):

        def setUp(self):

            self.ba = 2.661
            self.ha = 0.205
            self.Ca = 0.605
            self.h_stringer = 1.6 * 10 ** (-2)

            self.coordinates = get_crossectional_coordinates(self.Ca, self.ha, self.h_stringer)

            self.N = 50

            self.crosssection = CrossSection(self.coordinates, transform=False)
            self.transformed_model = FullModel(self.coordinates, (10e-10, self.ba), self.N, transform=True)
            self.normal_model = FullModel(self.coordinates, (10e-10, self.ba), self.N, transform=False)

        #@unittest.skip
        def test_boom_area_plot(self):

            Mx, My, Mz = moment_functions(distance_dict, force_dict)

            # for section in self.transformed_model:
            #     calculations = NumericalBending(section)
            #     stresses = calculations.calculate_bending_stress(Mz, My)
            #     calculations.boom_areas_calculator(stresses)

            #for section in self.normal_model:
            calculations = NumericalBending(self.normal_model[0])
            stresses = calculations.calculate_bending_stress(Mz, My)
            calculations.boom_areas_calculator(stresses)

            # for section in self.normal_model:
            #     for boom in section:
            #         print(boom)

            fig = plt.figure()
            ax = Axes3D(fig)

            #self.transformed_model.plot_structure(fig, ax, scaled_sizes=True)
            self.normal_model.plot_structure(fig, ax, scaled_sizes=True)

    def run_TestCases():
        suite = unittest.TestLoader().loadTestsFromTestCase(BendingTestCases)
        unittest.TextTestRunner(verbosity=2).run(suite)

    run_TestCases()
