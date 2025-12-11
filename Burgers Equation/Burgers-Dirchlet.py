# dirichlet
# euler

import numpy as np
import matplotlib.pyplot as plt

T = 2
dt = 0.00031
N = int(T/dt) + 1
mu = 0.1

L = np.pi
nx = 700
x = np.linspace(-L, L, nx)
dx = 2*L/(nx-1)

t = np.zeros(N)
u = np.zeros((N, nx))


#y0 = np.sin(np.pi*x/L)
#y0 = np.exp(-10*(x-L/2)**2)
#u0 = np.cos((np.pi/10*(x-L/2)))
u0 = -np.sin(2*x) #+ 1/2*np.sin(4*np.pi*x) + 1/3*np.sin(6*np.pi*x)

u[0] = u0

c1 = u0[0]
c2 = u0[-1]


def laplacian1D(vec):

    newvec = np.zeros_like(vec)

    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:])/dx**2

    return newvec



for i in range(N-1):
    t[i+1] = t[i] + dt
    u[i+1] = u[i] + mu*laplacian1D(u[i])*dt - u[i]*np.gradient(u[i], dx)*dt 

    u[i+1,0] = c1
    u[i+1,-1] = c2




#print(np.gradient(u0))

from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

fig, ax = plt.subplots()


ax.grid(True, color='grey', linestyle='-', linewidth=0.2)
plt.xlabel('x')
plt.ylabel('u')

line, = ax.plot(x, u0)


ax.set_xlim(-L, L)
ax.set_ylim(np.min(u), np.max(u))

steps = 100

def update(frame):
    line.set_ydata(u[frame])

frames = range(0, N, steps)

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()