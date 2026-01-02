import numpy as np
import matplotlib.pyplot as plt




T = 365*24*3600*2.5
dt = 3600
N = int(T/dt) + 1

G = 6.6734810e-11 # m^3 / (kg s^2)

m1 = 2e30 # kg
m2 = 2e30
m3 = 2e30

au = 1.5e11


r1_0 = [-au, 0]
r2_0 = [au, 0]
r3_0 = [0, au]

v1_0 = [0, 1e4]
v2_0 = [0, 2e4]
v3_0 = [0, 5e4]



r1 = np.zeros((N,2))
r2 = np.zeros((N,2))
r3 = np.zeros((N,2))

v1 = np.zeros((N,2))
v2 = np.zeros((N,2)) 
v3 = np.zeros((N,2))


r1[0] = r1_0
r2[0] = r2_0
r3[0] = r3_0

v1[0] = v1_0
v2[0] = v2_0
v3[0] = v3_0

norm = np.linalg.norm

def gravitational_acceleration(r1, r2, r3, m1=m1, m2=m2, m3=m3):

    a1 = G*m2*(r2 - r1)/(norm(r2 - r1)**3) + G*m3*(r3 - r1)/(norm(r3 - r1)**3) # total acceleration experienced by first body
    a2 = G*m1*(r1 - r2)/(norm(r1 - r2)**3) + G*m3*(r3 - r2)/(norm(r3 - r2)**3) # total acceleration experienced by second body
    a3 = G*m1*(r1 - r3)/(norm(r1 - r3)**3) + G*m2*(r2 - r3)/(norm(r2 - r3)**3) # total acceleration experienced by third body

    return a1, a2, a3

a = gravitational_acceleration

for i in range(N-1):

    a1, a2, a3 = a(r1[i], r2[i], r3[i])

    r1[i+1] = r1[i] + v1[i] * dt + 1/2 * a1 * dt**2
    r2[i+1] = r2[i] + v2[i] * dt + 1/2 * a2 * dt**2
    r3[i+1] = r3[i] + v3[i] * dt + 1/2 * a3 * dt**2

    a1_next, a2_next, a3_next = a(r1[i+1], r2[i+1], r3[i+1])

    v1[i+1] = v1[i] + 1/2 * ( a1 + a1_next ) * dt
    v2[i+1] = v2[i] + 1/2 * ( a2 + a2_next ) * dt
    v3[i+1] = v3[i] + 1/2 * ( a3 + a3_next ) * dt


from matplotlib.animation import FuncAnimation, FFMpegWriter


plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(8,8))

ax.set_aspect('equal', 'box')

ax.set_xlim(-1.5*au, 1.5*au)
ax.set_ylim(0, 10*au)

(line1,) = ax.plot([], [])
(line2,) = ax.plot([], [])
(line3,) = ax.plot([], [])

(endpoint1,) = ax.plot([], [], marker='o', markersize=6)
(endpoint2,) = ax.plot([], [], marker='o', markersize=6)
(endpoint3,) = ax.plot([], [], marker='o', markersize=6)


steps = 10

frames = range(0, N, steps)

def update(frame):
    line1.set_data(r1[:frame,0], r1[:frame,1])
    line2.set_data(r2[:frame,0], r2[:frame,1])
    line3.set_data(r3[:frame,0], r3[:frame,1])

    endpoint1.set_data([r1[frame,0]], [r1[frame,1]])
    endpoint2.set_data([r2[frame,0]], [r2[frame,1]])
    endpoint3.set_data([r3[frame,0]], [r3[frame,1]])

    return line1, line2, line3, endpoint1, endpoint2, endpoint3


ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()

writer = FFMpegWriter(fps=30)
ani.save("3-body.mp4", writer=writer)