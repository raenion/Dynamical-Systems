"""
Lorenz system simulation using a 4th-order Runge–Kutta method.

Description:
Numerical integration and 3D visualization of the Lorenz attractor.

Notes:
- all values stored for analysis projects
"""

import numpy as np
import matplotlib.pyplot as plt

# Time:

T = 35
dt = 0.001
N = int(T/dt) + 1

# Space:

x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

# System parameters:

## Well-known butterfly attractor parameters:

sigma = 10
rho = 28
beta = 8/3

# Initialization:

x[0] = 0
y[0] = 2
z[0] = 20

# Runge-Kutta 4 integration scheme: 

for i in range(N-1):

    """
    Note in the following that we have to compute all of the slope estimates 
    by grouping them as follows:
    
    1. k1x, k1y, k1z
    2. k2x, k2y, k2z
    3. k3x, k3y, k3z
    4. k4x, k4y, k4z

    We CANNOT treat our spatial variables in succession by grouping them together:

    1. k1x, k2x, k3x, k4x
    2. k1y, k2y, k3y, k4y
    3. k1z, k2z, k3z, k4z

    as each slope estimate (barring the k1 starting estimates) is a function of 
    all of the previous slope estimates.
    
    """

    k1x = sigma * (y[i] - x[i])
    k1y = x[i]*(rho - z[i]) - y[i]
    k1z = x[i]*y[i] - beta*z[i]

    k2x = sigma * (y[i] + k1y*dt/2 - x[i] - k1x*dt/2)
    k2y = (x[i] + k1x*dt/2)*(rho - z[i] - k1z*dt/2) - y[i] - k1y*dt/2
    k2z = (x[i] + k1x*dt/2)*(y[i] + k1y*dt/2) - beta*(z[i] + k1z*dt/2)

    k3x = sigma * (y[i] + k2y*dt/2 - x[i] - k2x*dt/2)
    k3y = (x[i] + k2x*dt/2)*(rho - z[i] - k2z*dt/2) - y[i] - k2y*dt/2
    k3z = (x[i] + k2x*dt/2)*(y[i] + k2y*dt/2) - beta*(z[i] + k2z*dt/2)

    k4x = sigma * (y[i] + k3y*dt - x[i] - k3x*dt)
    k4y = (x[i] + k3x*dt)*(rho - z[i] - k3z*dt) - y[i] - k3y*dt
    k4z = (x[i] + k3x*dt)*(y[i] + k3y*dt) - beta*(z[i] + k3z*dt)

    x[i+1] = x[i] + (k1x + 2*k2x + 2*k3x + k4x)*dt/6

    y[i+1] = y[i] + (k1y + 2*k2y + 2*k3y + k4y)*dt/6
    
    z[i+1] = z[i] + (k1z + 2*k2z + 2*k3z + k4z)*dt/6

# Plotting:

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')

ax1.grid(False)

ax1.plot3D(x, y, z, color='black', linewidth=0.1)

plt.show()






