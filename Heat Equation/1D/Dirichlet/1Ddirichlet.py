import numpy as np
import matplotlib.pyplot as plt

"""
1D Heat Equation
Domain: [-1, 1]
BC: Dirichlet
Scheme: Explicit Euler (FTCS)
"""

# Space:

nx = 600
dx = 2/(nx-1)
x = np.linspace(-1, 1, nx)

# Time:

T = 3
dt = 0.00001
N = int(T/dt) + 1
t = np.zeros(N)

# Physics:

nu = 0.1

# Stability check and conditional adjustment:

if dt > dx**2/(2*nu):
    dt = dx**2/(2*nu)
    N = int(T/dt)+1
    t = np.zeros(N)
    print(f'Stability Warning: Timestep adjusted to dt = {dt}')

# Some possible initial conditions:

#u0 = 1/8*(x+1)**4
#u0 = np.sin(4*x)
#u0 = np.exp(100*x)
u0 = np.tan(1.5*x) # intentionally sharp IC to test stability

# Initialization: 

u = np.zeros((N, nx))
u[0] = u0

## Save boundary values as constants: 

c1 = u0[0]
c2 = u0[-1]

# Laplacian: 

def laplacian1D(vec):

    # Dirichlet boundary conditions respected by keeping boundaries 0
    
    newvec = np.zeros_like(vec) 
    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:]) / dx**2
    
    return newvec

# Explicit Euler scheme integrator loop:

for i in range(N-1):
    t[i+1] = t[i] + dt
    u[i+1] = u[i] + nu*laplacian1D(u[i])*dt

    # Keeping Dirichlet condition by setting boundaries back to fixed values:
    
    u[i+1, 0] = c1
    u[i+1, -1] = c2

# Plotting:

fig, ax = plt.subplots(figsize=(12,7))

plottimes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.5, 3]

for time in plottimes:
    bestindex = np.argmin(np.abs(t - time))
    ax.plot(x, u[bestindex], label=f't = {time:.2f}')

plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('black') 
plt.tick_params(colors='white')
plt.xlabel('x', color='w')
plt.ylabel('u', color='w')
plt.grid(True)
plt.legend()
plt.show()
