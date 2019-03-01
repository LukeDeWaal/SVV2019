import numpy as np
import matplotlib.pyplot as plt

from src.Shear_flow import shear_flow
import src.AppliedForcesMoments as AFM
from src.StructuralCalculations.Bending import *
from src.AppliedForcesMoments import moment_functions
from src.NumericalTools import AOT_rotation

#function to get the vertical displacement at any point along the x axis
def Displacement(R1y, R2y, R3y, P1, P2, q, theta, x):

    import numpy as np
    #Constants
    E = 68.9e9 # [Pa]
    Izz = 1.8e-5 # [m^4]
    A = 19475.24 # []
    B = -15986.23 # []
    x1 = 0.172  # [m]
    x2 = 1.211  # [m]
    x3 = 2.591  # [m]
    xa = 0.350  # [m]
    la = 2.661  # [m]
    theta = np.deg2rad(theta)  #degrees
    #Forces
    q = q
    P1 = P1
    P2 = P2
    
    #Error testing
    if x > la:
        raise ValueError('x Value is greater than the length of the aileron')
    elif x < 0:
        raise ValueError('x Value is smaller than 0')
        
    #Step Function
    elif x < x1:
        d = (0 + 0 + 0 + 0  +(A*x) + B - (0) - (q*np.cos(theta)*(x**4)/24)  ) / (-E*Izz)
    elif x < (x2-(xa/2)):
        d = ((R1y*((x-x1)**3)/6) + 0 + 0 + 0  + (A*x) + B - 0  - (q*np.cos(theta)*(x**4)/24)  ) / (-E*Izz)
    elif x < x2:
        d = ((R1y*((x-x1)**3)/6) + 0 + 0 + (P1*(np.sin(theta)*(x-(x2-(xa/2)))**3)/6)  + (A*x) + B - 0 - (q*np.cos(theta)*(x**4)/24)  ) / (-E*Izz)
    elif x < (x2+(xa/2)):
        d = ((R1y*((x-x1)**3)/6) + (R2y*((x-x2)**3)/6) + 0 + (P1*(np.sin(theta)*(x-(x2-(xa/2)))**3)/6)  + (A*x) + B - 0 - (q*np.cos(theta)*(x**4)/24)  ) / (-E*Izz)
    elif x < x3:
        d = ((R1y*((x-x1)**3)/6) + (R2y*((x-x2)**3)/6) + 0 + (P1*(np.sin(theta)*(x-(x2-(xa/2)))**3)/6)  + (A*x) + B - ((P2*np.sin(theta)*(x-(x2+(xa/2)))**3)/6)  - (q*np.cos(theta)*(x**4)/24)  ) / (-E*Izz)
    elif x <= la:
        d = ((R1y*((x-x1)**3)/6) + (R2y*((x-x2)**3)/6) + (R3y*((x-x3)**3)/6) + (P1*(np.sin(theta)*(x-(x2-(xa/2)))**3)/6)  + (A*x) + B - ((P2*np.sin(theta)*(x-(x2+(xa/2)))**3)/6)  - (q*np.cos(theta)*(x**4)/24)  ) / (-E*Izz)  
    #Returning the displacement (local)
    return round(d, 5)


#setting up the specification of the aileron
ba = 2.661
ha = 0.205
Ca = 0.605
h_stringer = 1.6 * 10 ** (-2)
N=100
R1y=AFM.force_dict['R1'][1]
R2y=AFM.force_dict['R2'][1]
R3y=AFM.force_dict['R3'][1]
P1=AFM.force_dict['Pi'][1]
P2=AFM.force_dict['Pii'][1]
q   = 5.54e3
theta=28

# this function calculates the angle of twist of a certain section by using the rate of twist of all the previous functions up to the point

def angle_twist(distance):

    #set up the full model
    coordinates = get_crossectional_coordinates(Ca, ha, h_stringer)
    normal_model = FullModel(coordinates, (10e-10, ba), N, transform=False)

    total_angle=0


    # determine the index of the location
    index=round(distance*N/AFM.distance_dict['ba'])

    #sum up the angles of twist up to that location
    for i, section in enumerate(normal_model[:index]):

        distance_sections=i*AFM.distance_dict['ba']/N
        total_angle=total_angle+ shear_flow(i*distance_sections)[1]*distance_sections

    #get the radii of the cross section of interest
    centroid = section.get_centroid()
    radii_section=[np.linalg.norm(coordinate - centroid) for coordinate in section.get_all_coordinates()]
    relative_position_y = [(coordinate - centroid)[1] for coordinate in section.get_all_coordinates()]
    relative_position_z = [(coordinate - centroid)[2] for coordinate in section.get_all_coordinates()]


    #return total angle of twist and array of radii from centroic
    return total_angle, np.array(radii_section), relative_position_y, relative_position_z, centroid, section.get_all_coordinates()

#set the distance of interest (in this case is rib C)
distance=float(AFM.distance_dict['x1']+AFM.distance_dict['xa']/2)

7
twist_angle_rib_C, radii, relative_position_y, relative_position_z, centroid, all_coordinates = angle_twist(distance)

#find the displacement of the booms by adding up the vertical due to shear and the one due to the twist
def total_displacement(y_displacement, angle_of_twist, radii, distance, relative_position_y, relative_position_z, centroid, all_coordinates):


    # luke check if the sign are correct but I think it should be fine
    # y_displacements=radii*np.sin(angle_of_twist)*np.sign(relative_position_z)*-1
    # z_displacements = radii * np.cos(angle_of_twist)*np.sign(relative_position_y)
    #
    new_coordinates = np.array([AOT_rotation(point=coordinate, angle=angle_of_twist, centroid=centroid) for coordinate in all_coordinates])

    y_displacements = np.array([new[1] - old[1] for new, old in zip(new_coordinates, all_coordinates)])
    z_displacements = np.array([new[2] - old[2] for new, old in zip(new_coordinates, all_coordinates)])
    print(y_displacement)
    #add the displacement due to bending
    y_displacements=-y_displacements+y_displacement

    displacements=[z_displacements, y_displacements]


    return displacements

a = total_displacement(Displacement(R1y, R2y, R3y, P1, P2, q, theta, distance), twist_angle_rib_C, radii, distance, relative_position_y, relative_position_z, centroid, all_coordinates)
print (a)


