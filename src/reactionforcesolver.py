def reaction_force_solver(theta, q, P2, d1y, d2y, d3y):
    import numpy as np
    #Setting up all constants
    #xn = spanwise location of hinges from origin
    x1 = 0.172  #m
    x2 = 1.211  #m
    x3 = 2.591  #m
    xa = 0.350  #m
    ha = 0.205  #m (height aileron)
    la = 2.661  #m (length aileron)
    C = 0.605   #m (chord length)
    theta = np.deg2rad(theta)  #degrees
    E = 68.91e9  #pascals (young modulus)
    Izz = 1.6e-5       # (moment of inertia)
    Iyy = 1.11e-4
    d1y = 11.54e-3   #m
    d2y = 0          #m
    d3y = 18.40e-3   #m
    q = q      #N
    P2 = P2     #N
    
    #Ax = b, where A is the matrix for all equations of motion, moment equations, and compatibility bending equations
    #0=variable not in equation, eg: Sum of forces in X => R2x (reaction @ hinge 2 in x) = 0
    #A, B, C, D = integration constants for compatibility equations
    #Naming of variables for inputs and final outputs
    namevar = np.array(["R3z", "R2z", "R1z", "R1y", "R2y", "R3y", "R2x", "P1", "A", "B", "C", "D"])
    nameeq = np.array(["SFx", "SFy", "SFz", "SM2y", "SM2z", "SM2x", "defy1", "defy2", "defy3", "defz1", "defz2", "defz3"])
    
    #    Sum of forces in X, Y, Z
    SFx = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    SFy = np.array([0, 0, 0, 1, 1, 1, 0, np.sin(theta), 0, 0, 0, 0])
    SFz = np.array([1, 1, 1, 0, 0, 0, 0, -np.cos(theta), 0, 0, 0, 0])
    
    #    Sum of moments around hinge 2 in Y, Z, X
    SM2y = np.array([-(x3-x2), 0, (x2-x1), 0, 0, 0, 0, -(np.cos(theta)*(xa/2)), 0, 0, 0, 0])
    SM2z = np.array([0, 0, 0, -(x2-x1), 0, (x3-x2), 0, -(np.sin(theta)*(xa/2)), 0, 0, 0, 0])
    #SM2x = np.array([0, 0, 0, 0, 0, 0, 0, (np.sin(theta)*(ha/2))-(np.cos(theta)*(ha/2)), 0, 0, 0, 0])
    SM2x = np.array([0, 0, 0, 0, 0, 0, 0, (ha/2), 0, 0, 0, 0])
    
    #    Deflection in Y
    defy1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, x1, 1, 0, 0])
    defy2 = np.array([0, 0, 0, ((x2-x1)**3)/6, 0, 0, 0, +(np.sin(theta)*(x2-(x2-(xa/2)))**3)/6, x2, 1, 0, 0])
    defy3 = np.array([0, 0, 0, ((x3-x1)**3)/6, ((x3-x2)**3)/6, 0, 0, +(np.sin(theta)*(x3-(x2-(xa/2)))**3)/6, x3, 1, 0, 0])
    
    #    Deflection in Z
    defz1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x1, 1])
    defz2 = np.array([0, 0, -((x2-x1)**3)/6, 0, 0, 0, 0, -np.cos(theta)*((x2-(x2-(xa/2)))**3)/6, 0, 0, x2, 1])
    defz3 = np.array([0, -((x3-x2)**3)/6, -((x3-x1)**3)/6, 0, 0, 0, 0, -np.cos(theta)*((x3-(x2-(xa/2)))**3)/6, 0, 0, x3, 1])
    
    #    Combining all equations into one matrix
    A = np.array([SFx, SFy, SFz, SM2y, SM2z, SM2x, defy1, defy2, defy3, defz1, defz2, defz3])
    
    #    "Answers" to equations
    B = np.array([(0), #SFx
                  (q*la*np.cos(theta))+(P2*np.sin(theta)), #SFy
                  (q*la*np.sin(theta))-(P2*np.cos(theta)), #SFz
    #              
                  (P2*np.cos(theta)*(xa/2))-(q*la*((la/2)-x2)*np.sin(theta)), #SM2y
                  (q*la*((la/2)-x2)*np.cos(theta))+(P2*np.sin(theta)*(xa/2)), #SM2z
    #              (q*la*np.cos(theta)*((C/4)-(ha/2)))-(P2*np.cos(theta)*(ha/2))+(P2*np.sin(theta)*(ha/2)), #SM2x
                  (q*la*((C/4)-(ha/2)))+(P2*(ha/2)), #SM2x
    #              
                  (-E*Izz*np.cos(theta)*d1y)+(q*np.cos(theta)*(x1**4)/24), #defyx1
                  (-E*Izz*np.cos(theta)*d2y)+(q*np.cos(theta)*(x2**4)/24), #defyx2
                  (-E*Izz*np.cos(theta)*d3y)+(q*np.cos(theta)*(x3**4)/24)+((P2*np.sin(theta)*(x3-(x2+(xa/2)))**3)/6), #defyx3
    #              
                  (-E*Iyy*np.sin(theta)*d1y)-((q*np.sin(theta)*(x1**4))/24), #defzx1
                  (-E*Iyy*np.cos(theta)*d2y)-((q*np.sin(theta)*(x2**4))/24), #defzx2
                  (-E*Iyy*np.cos(theta)*d3y)-((q*np.sin(theta)*(x3**4))/24)-((P2*np.cos(theta)*(x3-(x2+(xa/2)))**3)/6) #defzx3
                  ])
    
    #    Solving linear system
    X = np.linalg.solve(A,B)
    
    #Print Solution
#    for i in range(len(namevar)):
#        print(namevar[i] + " = " + str(round(X[i],2)))
    #    print(nameeq[i] + " = " + str(round(sum((A[i]*X))-int(B[i]))))
    
    output = np.array([A, namevar, B, nameeq, X, np.allclose(np.dot(A, X), B)])
    return output
