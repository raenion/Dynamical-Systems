# saving values
# Domain [0, Lx]x[0, Ly]
# di

import numpy as np


T = 3
dt = 0.0001
N = int(T/dt) + 1
c = 150

Lx = 3*np.pi
Ly = 3*np.pi
nx = 80
ny = 80

dx = Lx/(nx-1)
dy = Ly/(ny-1)

xvec = np.linspace(0, Lx, nx)
yvec = np.linspace(0, Ly, ny)

x,y = np.meshgrid(xvec, yvec)

t = np.zeros(N)
z = np.zeros((N, ny, nx))

#z0 = np.zeros((ny,nx)) + 0.3*np.sin(x)*np.sin(y)
z0 = np.exp(-((x - Lx/2)**2 + (y - Ly/2)**2))

# add cool initial conditions 

z[0] = z0

z[0, 0, :] = 0
z[0, -1, :] = 0
z[0, :, 0] = 0
z[0, :, -1] = 0

v0 = np.zeros((ny, nx)) #+ 0.1   # initial velocity

z[1] = z0 #+ v0*dt

def laplacian2D(mat):

    vlaplacian = np.zeros_like(mat)
    hlaplacian = np.zeros_like(mat)

    vlaplacian[1:-1, 1:-1] = (mat[0:-2, 1:-1] - 2*mat[1:-1, 1:-1] + mat[2:, 1:-1]) / dy**2
    hlaplacian[1:-1, 1:-1] = (mat[1:-1, 0:-2] - 2*mat[1:-1, 1:-1] + mat[1:-1, 2:]) / dx**2

    return vlaplacian + hlaplacian


for i in range(1,N-1):
    t[i] = t[i-1] + dt
    z[i+1] = 2*z[i] - z[i-1] + (c**2)*laplacian2D(z[i]) * (dt**2)
    z[i+1, 0, :] = 0
    z[i+1, -1, :] = 0
    z[i+1, :, 0] = 0
    z[i+1, :, -1] = 0


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12,8))

surf = ax.plot_surface(x,y,z0)

ax.set_zlim(-1,1)      


def update(frame):
    global surf 
    surf.remove()
    surf = ax.plot_surface(x,y,z[frame], cmap='inferno', vmin = -0.3, vmax = 0.3, edgecolor='k', linewidth=0.1)

    return surf,

steps = 10

frames = range(0, N, steps)

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()


