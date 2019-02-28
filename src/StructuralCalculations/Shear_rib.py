import numpy as np
from src.Structure import get_crossectional_coordinates, CrossSection
from src.Shear_flow import shear_flow
from src.StructuralCalculations.von_mises_stress import shear_stress
import src.AppliedForcesMoments as AFM

"""
Shear Calculations
"""


class NumericalShear_Rib:

    def __init__(self, cross_section):
        self.__cross_section = cross_section

    #the input of the shear skin is the shear skin without counting the shear in spar
    def rib_shear(self, shear_skin):

        ''' inputs of the functions are the following:
        :param shear in skin
        :param Cross Section function
        '''

        #consider only the three most important booms and sum shear flows


        #angle of inclination of back airfoil
        theta=15

        part_i_shear=np.arange(0,4)
        part_ii_shear=np.arange(4,17)

        part_i_front=np.arange(1,5)
        part_i_back = np.arange(0, 4)

        part_ii_front = np.array([5,6,7,8,9,10,11,12,13,14,15,16,0])
        part_ii_back = np.arange(4, 17)


        all_booms_coordinates=np.array([boom.get_position() for boom in self.__cross_section])


        '''Section 1'''

        #the total vertical force borne by the first section

        total_vertical_forces_1=shear_skin[part_i_shear] * np.absolute(all_booms_coordinates[:,1][part_i_front]-all_booms_coordinates[:,1][part_i_back])

        # the total horizontal force borne by the first section

        total_horizontal_forces_1 = shear_skin[part_i_shear] * np.absolute(all_booms_coordinates[:, 2][part_i_front] - all_booms_coordinates[:,2][part_i_back])

        total_vertical_force_1=np.sum(total_vertical_forces_1)
        total_horizontal_force_1=np.sum(total_horizontal_forces_1)

        #horizontal force going into the section by sum of the moments at lower spar cap

        moment_vertical_shear1=-total_vertical_forces_1 * np.absolute(all_booms_coordinates[:,1][part_i_front]- self.__cross_section.get_sparcap_coordinates()[1][1])

        moment_horizontal_shear1=total_horizontal_forces_1 * np.sign(all_booms_coordinates[:,1][part_i_front]-all_booms_coordinates[:,1][part_i_back])*np.absolute(all_booms_coordinates[:, 2][part_i_front] - self.__cross_section.get_sparcap_coordinates()[1][2])

        p_top1= np.sum(moment_vertical_shear1+moment_horizontal_shear1)/(self.__cross_section.get_sparcap_coordinates()[0][1]-self.__cross_section.get_sparcap_coordinates()[1][1])

        p_bottom1=total_horizontal_force_1-p_top1

        q_web_1=total_vertical_force_1/(self.__cross_section.get_sparcap_coordinates()[0][1]-self.__cross_section.get_sparcap_coordinates()[1][1])


        '''Section 2'''

        # the total vertical force borne by the second section

        total_vertical_forces_2 = shear_skin[part_ii_shear] * np.absolute(
            all_booms_coordinates[:, 1][part_ii_front] - all_booms_coordinates[:, 1][part_ii_back])

        # the total horizontal force borne by the second section
        total_horizontal_forces_2 = shear_skin[part_ii_shear] * np.absolute(
            all_booms_coordinates[:, 2][part_ii_front] - all_booms_coordinates[:, 2][part_ii_back])

        total_vertical_force_2 = np.sum(total_vertical_forces_2)
        total_horizontal_force_2 = np.sum(total_horizontal_forces_2)

        # horizontal force going into the section by sum of the moments at lower spar cap

        moment_vertical_shear2 = -total_vertical_forces_2 * np.absolute(
            all_booms_coordinates[:, 1][part_ii_front] - self.__cross_section.get_sparcap_coordinates()[1][1])

        moment_horizontal_shear2 = total_horizontal_forces_2 * np.sign(
            all_booms_coordinates[:, 1][part_ii_front] - all_booms_coordinates[:, 1][part_ii_back]) * np.absolute(
            all_booms_coordinates[:, 2][part_ii_front] - self.__cross_section.get_sparcap_coordinates()[1][2])

        p_top2_z = -np.sum(moment_vertical_shear2 + moment_horizontal_shear2) / (
                self.__cross_section.get_sparcap_coordinates()[0][1] - self.__cross_section.get_sparcap_coordinates()[1][1])

        p_top2_y=p_top2_z*np.tan(theta)

        p_bottom2_z = -(total_horizontal_force_2 + p_top2_z)

        p_bottom2_y = p_bottom2_z*np.tan(theta)

        q_web_2 = -(-total_vertical_force_2+p_bottom2_y-p_top2_y) / (
                self.__cross_section.get_sparcap_coordinates()[0][1] - self.__cross_section.get_sparcap_coordinates()[1][1])



        return q_web_1, q_web_2

Ca = 0.605
ha = 0.205
h_stringer = 1.6*10**(-2)
coordinates = get_crossectional_coordinates(Ca, ha, h_stringer)
crosssection = CrossSection(coordinates, )

shear_stresses_C=shear_stress(shear_flow(AFM.distance_dict['x2']+AFM.distance_dict['xa']/2)[0])
q_web_1, q_web_2=NumericalShear_Rib(crosssection).rib_shear(np.ones(17))

print ('q_web_1', q_web_1)
print ('q_web_2', q_web_2)


