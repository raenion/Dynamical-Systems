import numpy as np
import matplotlib.pyplot as plt

"""
2D Heat Equation
Domain: [-1, 1]x[-1, 1]
BC: Periodic
Scheme: Explicit Euler (FTCS)
"""

# Space:

nx = 50
ny = 50
dx = 2/(nx-1)
dy = 2/(ny-1)
xvec = np.linspace(-1, 1,  nx)
yvec = np.linspace(-1, 1, ny) 
x, y = np.meshgrid(xvec, yvec)

# Time:

T = 1
dt = 0.0001
N = int(T/dt) + 1
t = np.zeros(N)

# Physics:

mu = 0.1

# Some possible initial conditions:

u0 = np.sin(2*np.pi*x) + np.cos(2*np.pi*y)
#u0 = np.sin(2*np.pi*x) + 2/3*np.cos(5*np.pi*x) + np.sin(2*np.pi*y) + 2/3*np.cos(5*np.pi*y)
#u0 = np.sin(np.pi*x*y)
#u0 = np.sin(2*np.pi*x) + 2/3*np.cos(5*np.pi*x)

# Initialization: 

u = np.zeros((N, ny, nx))
u[0,:,:] = u0

# 2D Laplacian:

def laplacian2D(mat):

    # np.roll for periodic boundary conditions

    vlaplacian = (np.roll(mat, -1, axis=0) - 2*mat + np.roll(mat, 1, axis=0))/dy**2
    
    hlaplacian = (np.roll(mat, -1, axis=1) - 2*mat + np.roll(mat, 1, axis=1))/dx**2

    return vlaplacian + hlaplacian

# Explicit Euler scheme integrator loop:

for i in range(N-1):

    t[i+1] = t[i] + dt

    u[i+1,:,:] = u[i,:,:] + mu*laplacian2D(u[i,:,:])*dt


# Animation:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12,7))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_zlim(np.min(u[0]), np.max(u[0])) 

surf = ax.plot_surface(x,y,u[0], cmap='viridis', edgecolor='none')

# TODO: There are currently stutters/mini-jumps/inconsistencies in the animation.

def update(frame):
    global surf
    surf.remove()
    surf = ax.plot_surface(x,y,u[frame], cmap='viridis', edgecolor='none')
    
    return surf,

steps = 30

frames = range(0, len(u), steps)

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()
