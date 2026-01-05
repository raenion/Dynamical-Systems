"""
1D Wave Equation Solver using Finite Differences

System: ∂²y/∂t² = c² ∂²y/∂x²

Scheme: Explicit central difference in time and space

Boundary conditions: Dirichlet (Fixed Ends)

"""


import numpy as np
import matplotlib.pyplot as plt

# Space:

L = 10
nx = 500
dx = L/(nx-1)
x = np.linspace(0, L, nx)

# Time:

T = 5
dt = 0.001
N = int(T/dt) + 1
t = np.zeros(N)

# Physics:

c = 2

# Stability:

if c * dt / dx > 1:
    raise ValueError(f'Stability condition violated: c*dt/dx must be <= 1. Suggested: dt = {dx/c}.')

# Solution initialization:

y = np.zeros((N, nx))

## Some initial conditions:

#y0 = np.sin(np.pi*x/L)
#y0 = np.exp(-10*(x-L/2)**2)
y0 = np.cos((np.pi/10*(x-L/2)))

## Initialization:

y[0] = y0

y[1] = y0 # Zero-velocity initial condition: y(t=0) = y(t=dt)


def laplacian1D(vec):
    
    newvec = np.zeros_like(vec)

    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:])/dx**2

    return newvec

# Solver:


for i in range(1, N-1):

    # Note that Dirichlet BCs are implicitly satisfied by this scheme and the definition of laplacian1D.
    
    t[i] = t[i-1] + dt
    
    y[i+1] = 2*y[i] - y[i-1] + (c**2)*laplacian1D(y[i])*(dt**2)

# Plotting:

fig, ax = plt.subplots()

plottimes = [0,1,2]

for time in plottimes:
    bestindex = np.argmin(np.abs(t - time))
    ax.plot(x, y[bestindex], label=f't = {time:.2f}')
    ax.legend()

plt.legend()

plt.show()
