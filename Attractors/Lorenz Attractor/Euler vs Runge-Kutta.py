import numpy as np
import matplotlib.pyplot as plt

# Time: 

T = 35
dt = 0.001
N = int(T/dt) + 1

# Space:

x_e = np.zeros(N)
y_e = np.zeros(N)
z_e = np.zeros(N)

x_rk = np.zeros(N)
y_rk = np.zeros(N)
z_rk = np.zeros(N)

# System parameters:

## Well-known butterfly attractor parameters:

sigma = 10
rho = 28
beta = 8/3

# Initialization:

initial_condition = [0, 2, 20]

x_e[0], y_e[0], z_e[0] = initial_condition

x_rk[0], y_rk[0], z_rk[0] = initial_condition

# Euler integration scheme for x_e, y_e, z_e:

for i in range(N-1):
    x_e[i+1] = x_e[i] + sigma*(y_e[i] - x_e[i]) * dt
    y_e[i+1] = y_e[i] + ( x_e[i]*(rho - z_e[i]) - y_e[i] ) * dt
    z_e[i+1] = z_e[i] + ( x_e[i]*y_e[i] - beta*z_e[i] ) * dt

# Runge-Kutta 4 integration scheme for x_rk, y_rk, z_rk:

for i in range(N-1):

    k1x = sigma * (y_rk[i] - x_rk[i])
    k1y = x_rk[i]*(rho - z_rk[i]) - y_rk[i]
    k1z = x_rk[i]*y_rk[i] - beta*z_rk[i]

    k2x = sigma * (y_rk[i] + k1y*dt/2 - x_rk[i] - k1x*dt/2)
    k2y = (x_rk[i] + k1x*dt/2)*(rho - z_rk[i] - k1z*dt/2) - y_rk[i] - k1y*dt/2
    k2z = (x_rk[i] + k1x*dt/2)*(y_rk[i] + k1y*dt/2) - beta*(z_rk[i] + k1z*dt/2)

    k3x = sigma * (y_rk[i] + k2y*dt/2 - x_rk[i] - k2x*dt/2)
    k3y = (x_rk[i] + k2x*dt/2)*(rho - z_rk[i] - k2z*dt/2) - y_rk[i] - k2y*dt/2
    k3z = (x_rk[i] + k2x*dt/2)*(y_rk[i] + k2y*dt/2) - beta*(z_rk[i] + k2z*dt/2)

    k4x = sigma * (y_rk[i] + k3y*dt - x_rk[i] - k3x*dt)
    k4y = (x_rk[i] + k3x*dt)*(rho - z_rk[i] - k3z*dt) - y_rk[i] - k3y*dt
    k4z = (x_rk[i] + k3x*dt)*(y_rk[i] + k3y*dt) - beta*(z_rk[i] + k3z*dt)

    x_rk[i+1] = x_rk[i] + (k1x + 2*k2x + 2*k3x + k4x)*dt/6

    y_rk[i+1] = y_rk[i] + (k1y + 2*k2y + 2*k3y + k4y)*dt/6
    
    z_rk[i+1] = z_rk[i] + (k1z + 2*k2z + 2*k3z + k4z)*dt/6

# Animation:

plt.style.use('dark_background')

fig = plt.figure(figsize=(14,7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.xaxis.set_pane_color((0,0,0,1))
ax1.yaxis.set_pane_color((0,0,0,1))
ax1.zaxis.set_pane_color((0,0,0,1))

ax2.xaxis.set_pane_color((0,0,0,1))
ax2.yaxis.set_pane_color((0,0,0,1))
ax2.zaxis.set_pane_color((0,0,0,1))

ax1.grid(False)
ax2.grid(False)

def set_xyz(line, x_new, y_new, z_new):

    line.set_data(x_new,y_new)
    line.set_3d_properties(z_new)

from matplotlib.animation import FuncAnimation

(eline,) = ax1.plot(x_e, y_e, z_e, color='white', linewidth=0.5)
(rkline,) = ax2.plot(x_rk, y_rk, z_rk, color='white', linewidth=0.5)


def update(frame):
    set_xyz(eline, x_e[:frame], y_e[:frame], z_e[:frame])
    set_xyz(rkline, x_rk[:frame], y_rk[:frame], z_rk[:frame])

    return eline, rkline

steps = 10 # animation frame stride

frames = range(0, N, steps)


ani = FuncAnimation(fig, update, frames=frames, interval=20)


plt.show()






