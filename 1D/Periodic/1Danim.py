import numpy as np
import matplotlib.pyplot as plt

nx = 100
dt = 0.0001
T = 1
dx = 2/(nx-1)
nu = 0.1

N = int(T/dt)+1

x = np.linspace(-1,1,nx)
#u0 = np.cos(2*np.pi*x) + 1/2*np.cos(4*np.pi*x)
u0 = np.sin(4*np.pi*x) + 1/3*np.sin(7*np.pi*x)

u = np.zeros((N,nx))
t = np.zeros(N)

u[0] = u0
t[0] = 0


def laplacian(f):
    return (np.roll(f,-1) - 2*f + np.roll(f, 1))/dx**2

for i in range(N-1):
    
    u[i+1] = u[i] + nu*laplacian(u[i])*dt
    
    t[i+1] = t[i] + dt


## Animate 

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

step = 10 # gap between frames is this many integrator time steps

umin = np.min(u)
umax = np.max(u)

ax.set_ylim(umin, umax)
ax.set_xlim(-1, 1)

line, = ax.plot([], [])


def update(frames):
    line.set_data(x, u[frames])

    return line,                  ## 

frames = range(0, len(u), step) # frames can be an iterable as well as int 

ani = FuncAnimation(fig, update, frames=frames, interval=20)


plt.show()



