def displacement_y(R1y, R2y, R3y, P1, P2, A, B, q, theta, x):
    import numpy as np
    
    #Constants
    E = 68.91e9 # [Pa]
    Izz = 1.6e-5       # (moment of inertia)
    A = A # []
    B = B # []
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