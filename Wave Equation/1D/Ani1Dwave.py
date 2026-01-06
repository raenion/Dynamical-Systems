"""
1D Wave Equation Solver using Finite Differences

System: ∂²y/∂t² = c² ∂²y/∂x²

Scheme: Explicit central difference in time and space

Boundary conditions: Dirichlet (Fixed Ends)

Notes:
    - animation
    - full stored solution (for analysis projects)
"""

import numpy as np
import matplotlib.pyplot as plt

# Space:

L = 20
nx = 100
x = np.linspace(0, L, nx)
dx = L/(nx-1)

# Time:

T = 20
dt = 0.01
N = int(T/dt) + 1

# Physics:

c = 1

# Stability check and conditional adjustment:

if dt > dx/c:
    dt = dx/c
    print(f'dt altered -> dt = {dt}')

# Solution initialization:

y = np.zeros((N, nx))

## Some initial conditions:

#y0 = 0.35*np.sin(2*np.pi*x/L)

#y0 = np.cos((x-L/2))

y0 = 0.1*np.exp(-0.1*(x-L/2)**2)

y0_x = np.gradient(y0, dx)   # spatial derivative of y0
v0 = -c * y0_x  

## Imposing ICs:

y[0] = y0

y[1] = y0 + dt*v0 # Choosing initial condition to trigger a right-travelling wave.

def laplacian1D(vec):
    
    newvec = np.zeros_like(vec)

    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:])/dx**2

    return newvec

# Solver:

for i in range(1, N-1):
    y[i+1] = 2*y[i] - y[i-1] + (c**2)*laplacian1D(y[i])*(dt**2)
    y[i+1,0] = 0
    y[i+1,-1] = 0

# Animation:

from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.grid(True, color='grey', linestyle='-', linewidth=0.2)
plt.xlabel('x')
plt.ylabel('y')

line, = ax.plot(x, y[0])

ax.set_xlim(0, L)
ax.set_ylim(-1, 1)

def update(frame):
    line.set_ydata(y[frame])

    return line,

steps = 10

frames = range(0,N, steps)

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()
