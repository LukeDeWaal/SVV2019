"""This code determined the shear flow between booms of a cross sectional airfoil which has
already been idealized as booms"""
"""written for SVV at TU Delft, faculty of Aerospace Enigneering, 2019"""

import Structure as scr 
import numpy as np
from ForceMomentObjects import *


def main():
    """Initlizes and runs program
    returns the shear flow between booms as an array"""
    
    #Initilises the other moduals and classes
    globs = scr.get_globals()
    boom_cords = scr.get_crossectional_coordinates(globs[1], globs[0], globs[-1])
    xsec = scr.CrossSection(boom_cords)
 
    #Gets the booms
    booms = xsec.get_all_booms() #Booms is a list with each instance of the boom class as an entry
    
    #split the booms up into sec 1 and sec 2
    boom_sec1, boom_sec2 = split_booms(booms)
    #print(boom_sec1[0].get_size())
    
    #create the force acting on the cross section
    
    """For test cases:
    1 = forces are 100N in both directions
    2 = 100N in positive y direction
    3 = 100N in positive z direction"""
    force_y, force_z = test_cases(1) #y is up, z is towards TE
    
    force_spar = Force(np.array([0, force_y, force_z]), np.array([boom_sec1[0].get_position()[0],0,boom_sec1[0].get_position()[2]]))
    '''force acting at middle of the spar'''
    
    #calculate the open shear flow in both sections - make independant of boom number
    Vshear_flows1 = calc_Vshear_flows(boom_sec1, force_spar, xsec)
    Vshear_flows2 = calc_Vshear_flows(boom_sec2, force_spar, xsec)
    
    #determine eq for q0s in both sections in sec 1 and sec 2 using sum of moments a the lowest spar boom
    mat_ordered = order_booms(Vshear_flows1, Vshear_flows2, boom_sec1, boom_sec2, xsec) #orderes the matrix nicer
    base_shear_flow1, base_shear_flow2, rotation  = det_base_shearflow(mat_ordered, force_spar)
    
    total_shearflows = combined_shearflow(base_shear_flow1, base_shear_flow2, mat_ordered)
    
    ############print stuff#############
    print("base shear flows \n")
    print ("base shear flow section 1 is")
    print(base_shear_flow1)
    print ("base shear flow section 2 is")
    print(base_shear_flow2)
    print('')
    print("variable shear flows \n")
    print ("variable shear flow in all sections is")
    for i in mat_ordered[:,3]: print(i)
    print('')
    print('Total shear flow through all sections is')
    print(total_shearflows)
    print("rotation is {}".format(rotation / (28e9)))
    
def split_booms(booms):
    """Given booms which is an array contained of sperate booms as a class instance, split the booms up in
    to 2 sections as per the multicell nature"""
    #for boom in booms:
        #print(boom)
    #print(len(booms))
    area1 = booms[:5]
    area2 = list(booms[4:])
    area2.append(booms[0])
    area2 = np.array(area2)
    #print(area2)
    
    return area1, area2

def calc_Vshear_flows(booms, F, xsec):
    """Given the booms (as an array) the forces acting in X and Y directions, the shearflow between the 
    booms is determined and returned"""
    #Cross section properties
    Izz, Iyy = xsec.area_MOI('z'), xsec.area_MOI('y')
    
    #Creates the array of shearflows which will be returned
    shear_flows = np.zeros(len(booms))
    #Forces
    Fx, Fy, Fz = F.get_force()
    
    #Caclulates the coeffcients ones to avoid waste
    coef1 = -1 * (Fz * Izz) / (Izz * Iyy)
    coef2 = -1 * (Fy * Iyy) / (Izz * Iyy)
    prev_shear_flow = 0 #as caclulations start at cut, initial shearflow is 0
    for i in range(len(shear_flows)):
        shear_flows[i] = prev_shear_flow + coef1 * (booms[i].get_size() * booms[i].get_position()[2]) + \
            coef2 * (booms[i].get_size() * booms[i].get_position()[1])
        prev_shear_flow = shear_flows[i]
    #print(shear_flows)
    return shear_flows
                                  
def det_base_shearflow(mat, force):
    """given the variable shear flows in the cross section, the boom positions and the cross section properties
    the base shearfloes are determined in each section"""
    #determines the areas of sec1 and sec 2
    #height at spar is given as 0.605m and cord lengh is 0.205m
    """matrix structure for solving
    A:
    
    C1, C2, 0
    D1, D2, D3
    E1, E2, E3
    
    b:
    A1
    A2
    A3
    
    solving for
    q_0,1
    q_0,2
    G d\theta / dx
    
    """
    #cross section properties
    Area_1 = (np.pi / 8) * (0.205) ** 2
    Area_2 = 0.205 * (0.605 - mat[0][0].get_position()[2])
    
    #Creat the final matricies to be solved
    A_mat = np.zeros((3, 3), dtype='object') #will be used when solving for constant shear flows
    b = np.zeros((3, 1)) #will be used when solving for constant shear flows
    
    #Boom 4 is the point at which moments are being summed
    A1 = 0
    for i in range(np.shape(mat)[0] - 1):
        r = mat[i][0].det_distance(mat[-1][0]) #vector from boom 4 to boom i
        #print(np.cross(r, force.get_force()))
        A1 += np.cross(r, force.get_force())[0]
    
    #add in the moments from the forces acting on the section
    A1 += np.cross(force.get_position() - mat[-1][0].get_position(), force.get_force())[0]

    A_mat[0] = np.array([2 * mat[0][-1][0], 2 * mat[5][-1][1], 0]) #C1, C2, 0
    b[0] = A1 * -1 #b1
    
    #for section 1
    D1, D2, D3 = 0, 0, 1
    A2, A3 = 0, 0
    E1, E2, E3 = 0, 0, 1
    
    for i in range(np.shape(mat)[0]):
        if mat[i][-2] == '1':
            A2 += mat[i][2] * mat[i][3] / mat[i][4] #q * ds / t
            D1 += mat[i][3] / mat[i][4] #ds/ t
            D2 += mat[i][3] / mat[i][4] #ds/ t
            
        if mat[i][-2] == '2':
            A3 += mat[i][2] * mat[i][3] / mat[i][4] #q * ds / t
            E1 += mat[i][3] / mat[i][4] #ds/ t
            E2 += mat[i][3] / mat[i][4] #ds/ t           
        
        elif mat[i][-2] == 'both':
            A2 += mat[i][2] * mat[i][3] / mat[i][4] #q * ds / t
            A3 -= mat[i][2] * mat[i][3] / mat[i][4] #q * ds / t. Negative as the shaerflow through the spar is defined as positive for section 1
            D1 += mat[i][3] / mat[i][4] #ds/ t
            D2 += mat[i][3] / mat[i][4] #ds/ t            
            E1 += mat[i][3] / mat[i][4] #ds/ t
            E2 += mat[i][3] / mat[i][4] #ds/ t 
    
    coef1 = 1 / (2 * Area_1)
    coef2 = 1 / (2 * Area_2)
    
    A_mat[1] = np.array([D1 * coef1, D2 * coef1, D3])
    A_mat[2] = np.array([E1 * coef2, E2 * coef2, E3])
    b[1] = A2 * -coef1
    b[2] = A3 * -coef2
    #print(b)
    #print(A_mat)
    
    x = np.linalg.solve(A_mat.astype(float), b.astype(float))
    
    #print(x)
    
    return x
    #pass

def order_booms(shear_flows1, shear_flows2, boom_sec1, boom_sec2, xsec):
    """ """
    #boom0, boom1, length, qvar, thickness, sec in, sec area
    mat = np.zeros((18, 7), dtype='object')
    #cross section properties
    
    A1 = (np.pi / 8) * (0.205) ** 2
    A2 = 0.205 * (0.605 - boom_sec1[0].get_position()[1])
    
    row_mat = 0
    for i in range(0, len(boom_sec1)-1, 1): #adds skin booms from area 1
        boom0 = boom_sec1[i]
        boom1 = boom_sec1[i+1]
        length = np.linalg.norm(boom1.get_position() - boom0.get_position())
        mat[row_mat] = np.array([boom0, boom1, length, shear_flows1[i], 1e-3, '1', (A1, 0)])
        row_mat += 1
    #print(mat)
    for i in range(0, len(boom_sec2) - 1, 1): #adds skin booms from area 2
        boom0 = boom_sec2[i]
        boom1 = boom_sec2[i+1]
        length = np.linalg.norm(boom1.get_position() - boom0.get_position())
        mat[row_mat] = np.array([boom0, boom1, length, shear_flows2[i], 1e-3, '2', (0, A2)])
        row_mat += 1
        #print(row_mat)
    #add the spar to the  matrix at the end, defined as going from boom9 to boom 3
    length = np.linalg.norm(boom_sec1[0].get_position() - boom_sec1[-1].get_position())
    mat[row_mat] = np.array([boom_sec1[-1], boom_sec1[0], length, shear_flows1[-1] - shear_flows2[-1], 2.8e-3, 'both',(A1, A2)])
    #shear flow through the spar is defined as positive for section A
    return mat
                            
    

def combined_shearflow(base_shear_flow1, base_shear_flow2, mat):
    """combines the shearflows and returns then as an array with the same order at shearflows in mat"""
    shear_flowsT = np.zeros((len(mat[:,0]), 1))
    for i in range(len(mat)):
        if mat[i][-2] == '1':
            shear_flowsT[i] = base_shear_flow1 + mat[i][3]
        if mat[i][-2] == '2':
            shear_flowsT[i] = base_shear_flow2 + mat[i][3]
        elif mat[i][-2] == 'both':
            shear_flowsT[i] = mat[i][3] + base_shear_flow1 - base_shear_flow2 #shearfloe through spar defined as +ve anticlockwise in section 1
    return shear_flowsT
            


def test_cases(case):
    """Runs test cases"""
    if case == 1:
        return 100, 100
    elif case == 2:
        return 100, 0
    elif case == 3:
        return 0, 100
    
#runs main()
main()
    
