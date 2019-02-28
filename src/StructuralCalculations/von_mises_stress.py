import numpy as np
import matplotlib.pyplot as plt

from src.Shear_flow import shear_flow
import src.AppliedForcesMoments as AFM
from src.StructuralCalculations.Bending import *
from src.AppliedForcesMoments import moment_functions


def shear_stress(shear_flow):

    ''':param t: thickness wing
       :param shear_flow: shear flow in a point

       :return shear stress'''

    t=0.0011

    stress=shear_flow*t

    return stress

def von_mises(sigma_xx,shear_xy):

    ''':param sigma_xx: normal stress caused by My and Mz
       :param shear_xy: shear stress

       :return von mises stress'''

    v_mises_stress=np.sqrt((sigma_xx)**2+3*shear_xy)

    return v_mises_stress

#calculating the shear stress at two important locations rib A and rib C (These were chosen since th FEM predicts highest stresses here)

shear_stresses_C=shear_stress(shear_flow(AFM.distance_dict['x2']+AFM.distance_dict['xa']/2)[0])
shear_stresses_A=shear_stress(shear_flow(AFM.distance_dict['x1'])[0])


#set the aileron specification
ba = 2.661
ha = 0.205
Ca = 0.605
h_stringer = 1.6 * 10 ** (-2)
N=100

#get the coordinates and the full model
coordinates = get_crossectional_coordinates(Ca, ha, h_stringer)
normal_model = FullModel(coordinates, (10e-10, ba), N, transform=False)

#find the moment Mx and My along the x axis
Mz=moment_functions(AFM.distance_dict,AFM.force_dict)[2]
My=moment_functions(AFM.distance_dict,AFM.force_dict)[1]

#determine the distance of rib A and C from the origin
distance_A=AFM.distance_dict['x2']+AFM.distance_dict['xa']/2
distance_C=AFM.distance_dict['x1']

#determine the corresponding index of rib A and C
index_A=round(distance_A/AFM.distance_dict['ba']*N)
index_C=round(distance_C/AFM.distance_dict['ba']*N)


#calculate the von mises stresses at rib location A and C per each boom
von_mises_stressC=von_mises(NumericalBending(normal_model[index_C]).calculate_bending_stress(Mz,My),shear_stresses_C)[0]
von_mises_stressA=von_mises(NumericalBending(normal_model[index_A]).calculate_bending_stress(Mz,My),shear_stresses_A)[0]
# i don't know why but it saves the von mises as many times as we have booms, so i take the first element of the array



#plot the stresses in order to make a comparison with the validation model
fig=plt.figure(figsize=(15,7))
# added conversion to kN/mm^2
ax=fig.add_subplot(211)
ax.set_title('Aileron stresses at rib C ULC1')
ribC=ax.scatter(normal_model[index_C].get_all_coordinates()[:,2], normal_model[index_C].get_all_coordinates()[:,1], c=von_mises_stressC*10**(-9))
cbar = plt.colorbar(ribC)
cbar.ax.set_ylabel('avg Von Mises Stress at rib C (kN/$mm^2$)', rotation=90)


ax=fig.add_subplot(212)
ax.set_title('Aileron stresses at rib A ULC1')
ribA=ax.scatter(normal_model[index_A].get_all_coordinates()[:,2], normal_model[index_A].get_all_coordinates()[:,1], c=von_mises_stressA*10**(-9))
cbar = plt.colorbar(ribA)
cbar.ax.set_ylabel('avg Von Mises Stress at rib A (kN/$mm^2$)', rotation=90)

plt.show()
distances=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]

twist=0
twists=[]
for distance in distances:
    twist=twist+shear_flow(distance)[1]*distance
    twists.append(twist)

plt.plot(twists)
plt.show()