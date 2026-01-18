"""
Solves the 1D viscous Burgers' equation:
    u_t + u u_x = μ u_xx

using:
- Explicit Euler time stepping
- Finite differences
- Dirichlet boundary conditions (fixed endpoint values)
- Animation of solution evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Space:

L = np.pi
nx = 700
x = np.linspace(-L, L, nx)
dx = 2 * L / (nx - 1)

# Time:

T = 2
dt = 0.00031
N = int(T / dt) + 1
t = np.zeros(N)

# Physics:

mu = 0.1

# Initialization:

u = np.zeros((N, nx))

## Some initial conditions:

# u0 = np.sin(np.pi * x / L)
# u0 = np.exp(-10 * (x - L/2)**2)
# u0 = np.cos((np.pi/10 * (x - L/2)))
u0 = -np.sin(2 * x) #+ 1/2*np.sin(4*np.pi*x) + 1/3*np.sin(6*np.pi*x)

u[0] = u0

# Dirichlet boundary values (fixed in time):

c_left = u0[0]
c_right = u0[-1]

# Differential operators:

def laplacian1D(vec):
    
    # Second-order central finite difference Laplacian.

    # Implictly Dirichlet aligned.
    
    newvec = np.zeros_like(vec)
    newvec[1:-1] = (vec[:-2] - 2 * vec[1:-1] + vec[2:]) / dx**2

    return newvec


def gradient1D(vec):

    # Central finite difference gradient.
    
    grad = np.zeros_like(vec)
    grad[1:-1] = (vec[2:] - vec[:-2]) / (2 * dx)

    return grad

# 1st order Euler solver loop:

for i in range(N - 1):

    ux = gradient1D(u[i])
    uxx = laplacian1D(u[i])

    u[i + 1] = u[i] + dt * (
        -u[i] * ux       # nonlinear advection
        + mu * uxx       # (linear) diffusion
    )

    # Enforce Dirichlet boundary conditions:

    u[i + 1, 0] = c_left
    u[i + 1, -1] = c_right

    t[i + 1] = t[i] + dt

# Animation:

plt.style.use("dark_background")

fig, ax = plt.subplots()
ax.grid(True, color="grey", linestyle="-", linewidth=0.2)

ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_xlim(-L, L)
ax.set_ylim(np.min(u), np.max(u))

line, = ax.plot(x, u[0])

steps = 10
frames = range(0, N, steps)

def update(frame):
    line.set_ydata(u[frame])
    return line,

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()
