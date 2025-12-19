# Earth around sun 

import numpy as np
import matplotlib.pyplot as plt



# we assume sun is fixed, i.e. immune to gravitational force of Earth
# due to conservation of angular momentum orbit lies in a plane, say the xy plane

T = 365*24*3600
dt = 3600
N = int(T/dt) + 1

G = 6.6734810e-11 # m^3 / (kg s^2)
m_sun = 1.989e30 # kg
R = 1.496e11 # m   (average radius of earth's orbit around sun)

r0_earth = [R, 0]
v0_earth = [0, 29.8e3]
r0_sun = [0, 0]

r_earth = np.zeros((N,2))
v_earth = np.zeros((N,2))

r_sun = np.zeros((N,2)) + r0_sun


r_earth[0] = r0_earth
v_earth[0] = v0_earth

norm = np.linalg.norm


def gravitational_acceleration(r1, r2, m2):

    return G*m2*(r2 - r1)/(norm(r2 - r1)**3)

a = gravitational_acceleration

for i in range(N-1):

    r_earth[i+1] = r_earth[i] + v_earth[i] * dt + 1/2*a(r_earth[i], r_sun[i], m_sun) * dt**2
    v_earth[i+1] = v_earth[i] + 1/2*(a(r_earth[i], r_sun[i], m_sun) + a(r_earth[i+1], r_sun[i+1], m_sun) ) * dt


fig, ax = plt.subplots(figsize=(8,8))

ax.set_aspect('equal', 'box')

ax.plot(r_earth[:,0], r_earth[:,1])

plt.show()


