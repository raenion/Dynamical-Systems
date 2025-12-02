
import numpy as np
import matplotlib.pyplot as plt

nx = 200
ny = 200
xvec = np.linspace(-1,1,nx)
yvec = np.linspace(-1,1,ny)
dx = 2/(nx-1)
dy = 2/(ny-1)
nu = 0.1
dt = min(dx**2 / (4*nu), dy**2 / (4*nu))

# Ensure stability condition is not violated by parameters:

if dt > min(dx**2 / (4*nu), dy**2 / (4*nu)):
    dt = min(dx**2 / (4*nu), dy**2 / (4*nu))
    print(f'dt altered -> dt = {dt}')

x,y = np.meshgrid(xvec,yvec)

#u0 = x+y
#u0 = np.sin(x**2 + y**2)
#u0 = np.exp(-20 * (x**2 + y**2))
#u0 = np.exp(-40 * ((x - 0.5)**2 + y**2)) + np.exp(-40 * ((x + 0.5)**2 + y**2))
#u0 = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
#u0 = np.sin(2*np.pi*3*x) * np.sin(2*np.pi*5*y)
#u0 = np.sin(3*np.pi*x)*np.cos(2*np.pi*y) + np.exp(-10*(x**2 + y**2)) + y**3
u0 = np.zeros((ny,nx)) + 20

def laplacian2D(mat):

    vlaplacian = np.zeros_like(mat)
    hlaplacian = np.zeros_like(mat)
    
    vlaplacian[1:-1,:] = (mat[:-2,:] - 2*mat[1:-1,:] + mat[2:,:]) / dy**2
    hlaplacian[:,1:-1] = (mat[:,:-2] - 2*mat[:,1:-1] + mat[:,2:]) / dx**2

    return vlaplacian + hlaplacian

## Animation

from matplotlib.animation import FuncAnimation

# Initialization:

ucurr = u0
t = 0

fig, ax = plt.subplots(figsize=(12,7))
im = ax.imshow(ucurr, extent=(-1, 1, -1, 1), origin='lower', cmap='jet', animated=True, vmin=0, vmax=100, interpolation='bilinear')

plt.colorbar(im, ax=ax)

stepsperframe = 30

def update(frames):
    global ucurr, t
    for _ in range(stepsperframe):
        unew = ucurr + nu*laplacian2D(ucurr)*dt
        t += dt
        ucurr = unew
        ucurr[0, :] = 100
        ucurr[-1, :] = 0
        ucurr[:, 0] = 0
        ucurr[:,-1] = 100
    im.set_array(ucurr)
    ax.set_title(f'Distribution at t = {t:.3f}')
    return [im]

ani = FuncAnimation(fig, update, frames=100, interval=20)

plt.show()
