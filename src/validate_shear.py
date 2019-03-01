##Imports
import Structure as scr
import numpy as np
import Shear_flow as sf


##Code
"""This code takes the shear flows and converts it to shear to verify, it then
can be compared to the shear force in the analytical model"""

def verify_shear(x_position):
    """This is the function which is called to give the shear in the section
    it returns an array of the shear forces acting on a section
    Input: The x position in question
    """
    shear_flows, __ = sf.shear_flow(x_position, 1)
    #shear_flows, __ = sf.test_cases(7)
    
    return main(x_position, shear_flows)

def main(x_position, shear_flows):
    """Runs and initlises the code"""
    
    #Gets the section parameters
    globs = scr.get_globals()
    boom_cords = scr.get_crossectional_coordinates(globs[1], globs[0], globs[-1])
    xsec = scr.CrossSection(boom_cords, x_coordinate=x_position, transform=False)
 
    #Gets the booms
    booms = xsec.get_all_booms() #Booms is a list with each instance of the boom class as an entry
    
    ordered_mat = order_data(shear_flows, booms)
    
    shear_forces = np.sum(ordered_mat[:,-1])
    
    #print stuff
    print("Shear in x is {} N".format(shear_forces[0]))
    print("Shear in y is {} N".format(shear_forces[1]))
    print("Shear in z is {} N".format(shear_forces[2]))
    
    return shear_forces
    
    
    
def order_data(shear_flows, booms):
    """Takes the inout data and orders it to make other code easier to use."""
    mat = np.zeros((len(shear_flows),6), dtype='object')
    #Array format:
    """Start boom, end boom, shear flow between booms, length between booms,
    unit vector going from start boom to end boom, shear force vector bewteen booms """
    

    for i in range(len(booms)-1):
        boom0 = booms[i]
        boom1 = booms[i+1]
        r_vec = boom1.get_position() - boom0.get_position()
        length = np.linalg.norm(r_vec)
        r_dir_vec = r_vec / length
        F_vec = length * r_dir_vec * shear_flows[i]
        mat[i] = np.array([boom0, boom1, shear_flows[i], length, r_dir_vec, F_vec])
        #print(boom0.get_label(), boom1.get_label(), shear_flows[i], length, r_dir_vec, F_vec)
    #for the last row
    r_vec = booms[0].get_position() - booms[4].get_position()
    length = np.linalg.norm(r_vec)
    mat[-1] = np.array([booms[4], booms[0], shear_flows[-1], length, \
                       r_vec / length, shear_flows[-1] * r_vec])
    
    return mat 
    
verify_shear(1)