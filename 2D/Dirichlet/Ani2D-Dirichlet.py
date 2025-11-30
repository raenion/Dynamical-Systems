import numpy as np
import matplotlib.pyplot as plt

T = 2
dt = 0.001
N = int(T/dt)+1
nx = 100
ny = 100
xvec = np.linspace(-1,1,nx)
yvec = np.linspace(-1,1,ny)
dx = 2/(nx-1)
dy = 2/(ny-1)
nu = 0.1

x,y = np.meshgrid(xvec,yvec)

t = np.zeros(N)
u = np.zeros((N, ny, nx))

u0 = np.sin(x*y)
u[0] = u0


# set boundary temperature for more flexibility for dirichlet enforcement (##)
# e.g. c1, c2, c3, c4 np arrays of length nx or ny, giving maximally general manipulation of dirchlet BCs


def laplacian2D(mat):

    vlaplacian = np.zeros_like(mat)
    hlaplacian = np.zeros_like(mat)
    
    vlaplacian[1:-1, 1:-1] = (mat[0:-2, 1:-1] - 2*mat[1:-1, 1:-1] + mat[2:, 1:-1]) / dy**2
    hlaplacian[1:-1, 1:-1] = (mat[1:-1, 0:-2] - 2*mat[1:-1, 1:-1] + mat[1:-1, 2:]) / dx**2

    return vlaplacian + hlaplacian


for i in range(N-1):
    t[i+1] = t[i] + dt
    u[i+1] = u[i] + nu*laplacian2D(u[i])*dt
    u[i+1, 0, :] = 0                                ##
    u[i+1, -1,:] = 0                                ##
    u[i+1, :,0] = 0                                 ##
    u[i+1, :,-1] = 0                                ##


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


plt.style.use('dark_background')

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12,7))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')


surf = ax.plot_surface(x,y,u[0], cmap='viridis', edgecolor='none')
ax.set_zlim(np.min(u), np.max(u))

def update(frames):
    global surf                                             # so that .remove doesnt think its trying to be applied to surf that is redefined after it?
    surf.remove()                                           # instead of ax.clear so update() doesnt have to redraw unnecessary things every time its called by funcanimation
    surf = ax.plot_surface(x,y,u[frames], cmap='viridis', edgecolor='none')
    return surf,

steps = 1

frames = range(0, len(u), steps)

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()





