import numpy as np
import matplotlib.pyplot as plt

"""
2D Heat Equation
Domain: [-1, 1]x[-1, 1]
BC: Dirichlet
Scheme: Explicit Euler (FTCS)
"""

# Space:

nx = 100
ny = 100
xvec = np.linspace(-1,1,nx)
yvec = np.linspace(-1,1,ny)
dx = 2/(nx-1)
dy = 2/(ny-1)
x, y = np.meshgrid(xvec, yvec)

# Time:

T = 2
dt = 0.001 # add autoset to stability bound for inappropriate values
N = int(T/dt) + 1
t = np.zeros(N)

# Physics:

nu = 0.1

# Initial Condition:

u0 = np.sin(np.pi*(x**2 + y**2))

# Initialization:

u = np.zeros((N, ny, nx))
u[0] = u0

# Storing initial boundaries to enforce Dirichlet conditions:

c1 = u[0, 0, :]
c2 = u[0, :, -1]
c3 = u[0, -1, :]
c4 = u[0, :, 0]

# 2D Laplacian, with zero-edges inkeeping with Dirichlet:

def laplacian2D(mat):

    vlaplacian = np.zeros_like(mat)
    hlaplacian = np.zeros_like(mat)
    
    vlaplacian[1:-1, 1:-1] = (mat[0:-2, 1:-1] - 2*mat[1:-1, 1:-1] + mat[2:, 1:-1]) / dy**2
    hlaplacian[1:-1, 1:-1] = (mat[1:-1, 0:-2] - 2*mat[1:-1, 1:-1] + mat[1:-1, 2:]) / dx**2

    return vlaplacian + hlaplacian

# Explicit Euler integration scheme 

for i in range(N-1):
    t[i+1] = t[i] + dt
    
    u[i+1] = u[i] + nu*laplacian2D(u[i])*dt
    
    u[i+1, 0, :] = c1
    u[i+1, :, -1] = c2
    u[i+1, -1, :] = c3
    u[i+1, :, 0] = c4

# Animation:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12,7))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')


surf = ax.plot_surface(x,y,u[0], cmap='viridis', edgecolor='none')
ax.set_zlim(np.min(u), np.max(u))

# animation currently taxing/stuttery: maybe find alternative to surface plot

def update(frames):
    global surf
    surf.remove()
    surf = ax.plot_surface(x,y,u[frames], cmap='viridis', edgecolor='none')
    return surf,

steps = 1

frames = range(0, len(u), steps)

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()





