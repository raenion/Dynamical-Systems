import numpy as np
import matplotlib.pyplot as plt

T = 1
dt = 0.0001
N = int(T/dt) + 1
nx = 50
ny = 50
dx = 2/(nx-1)
dy = 2/(ny-1)
mu = 0.1

xvec = np.linspace(-1, 1,  nx)
yvec = np.linspace(-1, 1, ny) 
x, y = np.meshgrid(xvec,yvec)

t = np.zeros(N)
u = np.zeros((N, ny, nx))


u0 = np.sin(2*np.pi*x) + np.cos(2*np.pi*y)
#u0 = np.sin(2*np.pi*x) + 2/3*np.cos(5*np.pi*x) + np.sin(2*np.pi*y) + 2/3*np.cos(5*np.pi*y)
#u0 = np.sin(np.pi*x*y)
#u0 = np.sin(2*np.pi*x) + 2/3*np.cos(5*np.pi*x)

u[0,:,:] = u0


def laplacian2D(mat):

    vlaplacian = (np.roll(mat, -1, axis=0) - 2*mat + np.roll(mat, 1, axis=0))/dy**2
    
    hlaplacian = (np.roll(mat, -1, axis=1) - 2*mat + np.roll(mat, 1, axis=1))/dx**2

    return vlaplacian + hlaplacian


for i in range(N-1):

    t[i+1] = t[i] + dt

    u[i+1,:,:] = u[i,:,:] + mu*laplacian2D(u[i,:,:])*dt


###### Animate

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


plt.style.use('dark_background')

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12,7))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')


surf = ax.plot_surface(x,y,u[0], cmap='viridis', edgecolor='none')

# there are currently stutters/mini-jumps/inconsistencies in the animation, why?

def update(frames):
    global surf                                             # so that .remove doesnt think its trying to be applied to surf that is redefined after it?
    surf.remove()                                           # instead of ax.clear so update() doesnt have to redraw unnecessary things every time its called by funcanimation
    surf = ax.plot_surface(x,y,u[frames], cmap='viridis', edgecolor='none')
    ax.set_zlim(np.min(u[0]), np.max(u[0]))                                     # doesnt need to be in the loop
    return surf,

steps = 30

frames = range(0, len(u), steps)

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()
