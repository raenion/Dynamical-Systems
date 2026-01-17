"""
Solves the 1D viscous Burgers' equation:
    u_t + u u_x = μ u_xx

using:
- Explicit Euler time stepping
- Second-order finite differences
- Dirichlet boundary conditions (implicit: u=0 at boundaries)
"""

import numpy as np
import matplotlib.pyplot as plt

# Space:

L = np.pi
nx = 200
x = np.linspace(-L, L, nx)
dx = 2*L/(nx-1)

# Time:

T = 5
dt = 0.001
N = int(T/dt) + 1
t = np.zeros(N)

# Physics:

mu = 0.1

# NOTE:
# Explicit Euler requires small dt for stability,
# especially when advection dominates diffusion.

assert dt <= dx**2/(2*mu), "Time step may be unstable"

# Initialization:

u = np.zeros((N, nx))

## Some initial conditions:

#y0 = np.sin(np.pi*x/L)
#y0 = np.exp(-10*(x-L/2)**2)
#u0 = np.cos((np.pi/10*(x-L/2)))
u0 = -np.sin(2*x) #+ 1/2*np.sin(4*np.pi*x) + 1/3*np.sin(6*np.pi*x)

u[0] = u0

def laplacian1D(vec):

    # Implicitly Dircihlet by keeping boundaries zero.

    newvec = np.zeros_like(vec)
    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:])/dx**2

    return newvec

def gradient1D(vec):

    grad = np.zeros_like(vec)
    grad[1:-1] = (vec[2:] - vec[:-2])/(2*dx)

    return grad

# 1st order Euler solver loop:

for i in range(N-1):

    ux = gradient1D(u[i])

    uxx = laplacian1D(u[i])

    u[i+1] = u[i] + dt * (
        - u[i] * ux     # nonlinear advection
        + mu * uxx      # linear diffusion
    )
    
    t[i+1] = t[i] + dt
    
# Plotting:

fig, ax = plt.subplots()

plottimes = [0, 0.25, 0.5, 1, 2, 3]

for time in plottimes:
    bestindex = np.argmin(np.abs(t - time))
    ax.plot(x, u[bestindex], label=f't = {time:.2f}')

ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
plt.legend()
plt.show()
