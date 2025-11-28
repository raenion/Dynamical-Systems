import numpy as np
import matplotlib.pyplot as plt

T = 3
dt = 0.00001
N = int(T/dt) + 1
nx = 600
dx = 2/(nx-1)
nu = 0.1

x = np.linspace(-1, 1, nx)
t = np.zeros(N)
u = np.zeros((N, nx))

#u0 = 1/8*(x+1)**4
#u0 = np.sin(4*x)
#u0 = np.exp(100*x)
u0 = np.tan(1.5*x)


u[0] = u0

c1 = u0[0]
c2 = u0[-1]

def laplacian1D(vec):      # same function name as periodic 1d laplacian
    
    newvec = np.zeros_like(vec) 
    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:]) / dx**2
    
    return newvec

for i in range(N-1):
    t[i+1] = t[i] + dt
    u[i+1] = u[i] + nu*laplacian1D(u[i])*dt
    u[i+1,0] = c1
    u[i+1,-1] = c2

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