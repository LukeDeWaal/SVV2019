import numpy as np
from reactionforcesolver import reaction_force_solver
from displacementfunction import displacement_y

#Testing Load Cases
theta = 28      # [degrees]
q = 5.54e3      # [N]
P2 = 97.4e3     # [N]
d1y = 11.54e-3  # [m]
d2y = 0         # [m]
d3y = 18.40e-3  # [m]

#Pulling the forces from the output of the reaction force solver
#LOCAL FORCES
R1y = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][3]
R1z = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][2]
R2y = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][4]
R2z = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][1]
R3y = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][5]
R3z = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][0]
P1 = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][7]
X = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4]

#Converting the local coordinate forces into the global reference
thetarad = np.deg2rad(theta)
print()
#GLOBAL FORCES
print("R1y = " + str(R1y*np.cos(thetarad)+R1z*np.sin(thetarad)) + " [N]")
print("R2y = " + str(R2y*np.cos(thetarad)+R2z*np.sin(thetarad)) + " [N]")
print("R3y = " + str(R3y*np.cos(thetarad)+R3z*np.sin(thetarad)) + " [N]")

print("R1z = " + str(R1z*np.cos(thetarad)-R1y*np.sin(thetarad)) + " [N]")
print("R2z = " + str(R2z*np.cos(thetarad)-R2y*np.sin(thetarad)) + " [N]")
print("R3z = " + str(R3z*np.cos(thetarad)-R3y*np.sin(thetarad)) + " [N]")
print("P1 = " + str(P1) + "[N]")

#Pulling the constants of integration from the reaction force solver
A = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][8]
B = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][9]
C = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][10]
D = reaction_force_solver(theta, q, P2, d1y, d2y, d3y)[4][11]

d1 = (displacement_y(R1y, R2y, R3y, P1, P2, A, B, q, theta, 0.172))
d2 = (displacement_y(R1y, R2y, R3y, P1, P2, A, B, q, theta, 1.211))
d3 = (displacement_y(R1y, R2y, R3y, P1, P2, A, B, q, theta, 2.591))

print()
print(str(round(d1/np.cos(thetarad), 5)) + " m")
print(str(round(d2/np.cos(thetarad), 5)) + " m")
print(str(round(d3/np.cos(thetarad), 5)) + " m")