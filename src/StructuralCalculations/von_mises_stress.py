import numpy as np

def shear_stress(shear_flow,t):

    ''':param t: thickness wing
       :param shear_flow: shear flow in a point

       :return shear stress'''

    stress=shear_flow*t

    return stress

def von_mises(sigma_xx,shear_xy):

    ''':param sigma_xx: normal stress caused by My and Mz
       :param shear_xy: shear stress

       :return von mises stress'''

    v_mises_stress=np.sqrt((sigma_xx)**2+3*shear_xy)

    return v_mises_stress