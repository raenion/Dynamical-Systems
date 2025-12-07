import numpy as np
import matplotlib.pyplot as plt



T = 20
dt = 0.01
N = int(T/dt) + 1
c = 1

L = 20
nx = 100
x = np.linspace(0, L, nx)
dx = L/(nx-1)

t = np.zeros(N)
y = np.zeros((N, nx))

if dt > dx/c:
    dt = dx/c
    print(f'dt altered -> dt = {dt}')


#y0 = 0.35*np.sin(2*np.pi*x/L)

#y0 = np.cos((x-L/2))

y0 = 0.1*np.exp(-0.1*(x-L/2)**2)

y0_x = np.gradient(y0, dx)   # derivative of u0
v0 = -c * y0_x  

y[0] = y0

y[1] = y0 #+ dt*v0



def laplacian1D(vec):
    newvec = np.zeros_like(vec)

    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:])/dx**2

    return newvec

for i in range(1, N-1):
    t[i] = t[i-1] + dt
    y[i+1] = 2*y[i] - y[i-1] + (c**2)*laplacian1D(y[i])*(dt**2)
    y[i+1,0] = 0
    y[i+1,-1] = 0



from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')


fig, ax = plt.subplots()

ax.grid(True, color='grey', linestyle='-', linewidth=0.2)
plt.xlabel('x')
plt.ylabel('y')

line, = ax.plot(x, y[0])

ax.set_xlim(0,L)
ax.set_ylim(-1, 1)

def update(frame):
    line.set_ydata(y[frame])

    return line,


steps = 10

frames = range(0,N, steps)

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()