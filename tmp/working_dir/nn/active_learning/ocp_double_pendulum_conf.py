import numpy as np

T = 0.5                   # OCP horizion
dt = 0.01               # OCP time step
max_iter = 100          # Maximum iteration per point

lowerPositionLimit1 = 3/4*np.pi
upperPositionLimit1 = 5/4*np.pi
lowerVelocityLimit1 = -10
upperVelocityLimit1 = 10
lowerControlBound1 = -9.81*3.5
upperControlBound1 = 9.81*3.5

lowerPositionLimit2 = 3/4*np.pi
upperPositionLimit2 = 5/4*np.pi
lowerVelocityLimit2 = -10
upperVelocityLimit2 = 10
lowerControlBound2 = -9.81
upperControlBound2 = 9.81

w_q1 = 1e2
w_v1 = 1e-1
w_u1 = 1e-4

w_q2 = 1e2
w_v2 = 1e-1
w_u2 = 1e-4