import numpy as np
import matplotlib.pyplot as plt

T = 10


nx = 40
ny = 40
xvec = np.linspace(-1,1,nx)
yvec = np.linspace(-1,1,ny)
dx = 2/(nx-1)
dy = 2/(ny-1)
nu = 1
dt = 0.00001

if dt > min(dx**2 / (4*nu), dy**2 / (4*nu)):
    dt = min(dx**2 / (4*nu), dy**2 / (4*nu))
    print(f'dt altered -> dt = {dt}')

N = int(T/dt)+1

x,y = np.meshgrid(xvec,yvec)

#u0 = x+y
#u0 = np.sin(x**2 + y**2)
#u0 = np.exp(-20 * (x**2 + y**2))
#u0 = np.exp(-40 * ((x - 0.5)**2 + y**2)) + np.exp(-40 * ((x + 0.5)**2 + y**2))
#u0 = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
#u0 = np.sin(2*np.pi*3*x) * np.sin(2*np.pi*5*y)
#u0 = np.sin(3*np.pi*x)*np.cos(2*np.pi*y) + np.exp(-10*(x**2 + y**2)) + y**3
u_now = np.zeros((ny,nx))

u_now[0,:] = 0
u_now[-1,:] = 100
u_now[:,0] = 0
u_now[:,-1] = 100


def laplacian2D(mat):

    vlaplacian = np.zeros_like(mat)
    hlaplacian = np.zeros_like(mat)
    
    vlaplacian[1:-1, 1:-1] = (mat[:-2, 1:-1] - 2*mat[1:-1, 1:-1] + mat[2:, 1:-1]) / dy**2
    hlaplacian[1:-1, 1:-1] = (mat[1:-1, :-2] - 2*mat[1:-1, 1:-1] + mat[1:-1, 2:]) / dx**2

    return vlaplacian + hlaplacian



fig, ax = plt.subplots(figsize=(7,5))
im = ax.imshow(u_now, extent=(-1,1,-1,1), origin='lower', cmap='jet', vmin=0, vmax=100) # can add bilinear interpolate

t = 0

steps = 100       # big while loop for large step count. could cause stuttery animation

count = 0
while t <= T:
    if count % steps == 0:
        im.set_array(u_now)
        ax.set_title(f't = {t:.3f}s')
        plt.pause(0.001)
    u_next = u_now + nu*laplacian2D(u_now)*dt
    t += dt
    u_now = u_next
    count += 1


plt.show()

print(dt)