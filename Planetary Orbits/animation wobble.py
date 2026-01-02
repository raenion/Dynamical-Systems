import numpy as np
import matplotlib.pyplot as plt


# ToDo: clean, sort, make planet more massive, move closer to star, rename variables

# we NO LONGER assume star is fixed, i.e. we compute gravitational pull of planet on star
# due to conservation of angular momentum orbit lies in a plane, say the xy plane

# Time:

T = 365*24*3600
dt = 360
N = int(T/dt) + 1

# Physics:

G = 6.6734810e-11 # m^3 / (kg s^2)
m_star = 1.989e30 # kg
m_planet = 1e28 # kg
R = 1e10 # m

r0_planet = [R, 0]
v0_planet = [0, 29.8e3]
r0_star = [0, 0]
v0_star = [0, 0]

r_planet = np.zeros((N,2))
v_planet = np.zeros((N,2))

r_star = np.zeros((N,2))
v_star = np.zeros((N,2)) 

r_planet[0] = r0_planet
v_planet[0] = v0_planet

r_star[0] = r0_star
v_star[0] = v0_star

norm = np.linalg.norm

def gravitational_acceleration(r1, r2, m2):

    return G*m2*(r2 - r1)/(norm(r2 - r1)**3)

a = gravitational_acceleration

# Verlet integration scheme:

for i in range(N-1):

    r_planet[i+1] = r_planet[i] + v_planet[i] * dt + 1/2*a(r_planet[i], r_star[i], m_star) * dt**2    
    r_star[i+1] = r_star[i] + v_star[i] * dt + 1/2*a(r_star[i], r_planet[i], m_planet) * dt**2

    v_planet[i+1] = v_planet[i] + 1/2*(a(r_planet[i], r_star[i], m_star) + a(r_planet[i+1], r_star[i+1], m_star) ) * dt
    v_star[i+1] = v_star[i] + 1/2*( a(r_star[i], r_planet[i], m_planet) + a(r_star[i+1], r_planet[i+1], m_planet) ) * dt

# Animation: 

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8,8))

(line_planet,) = ax.plot([], [])
(line_star,) = ax.plot([], [])

ax.set_ylim(-1.5*R, 1.5*R)
ax.set_xlim(-1.5*R, 1.5*R)

steps = 10

frames = range(0, N, steps)

def update(frame):
    line_planet.set_data(r_planet[:frame, 0], r_planet[:frame, 1])
    line_star.set_data(r_star[:frame, 0], r_star[:frame, 1])

    return line_planet, line_star


ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()


