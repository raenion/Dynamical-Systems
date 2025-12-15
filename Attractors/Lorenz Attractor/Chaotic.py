import numpy as np
import matplotlib.pyplot as plt

norm = np.linalg.norm

T = 35
dt = 0.001
N = int(T/dt) + 1

sigma = 10
rho = 28
beta = 8/3

x1 = np.zeros(N)
y1 = np.zeros(N)
z1 = np.zeros(N)

x2 = np.zeros(N)
y2 = np.zeros(N)
z2 = np.zeros(N)

t = np.zeros(N)

initial_condition1 = [15, 3, 10]
initial_condition2 = [15, 3, 10.1]


x1[0], y1[0], z1[0] = initial_condition1

x2[0], y2[0], z2[0] = initial_condition2


for x,y,z in zip([x1,x2], [y1, y2], [z1, z2]):


    for i in range(N-1):

        k1x = sigma * (y[i] - x[i])
        k1y = x[i]*(rho - z[i]) - y[i]
        k1z = x[i]*y[i] - beta*z[i]

        k2x = sigma * (y[i] + k1y*dt/2 - x[i] - k1x*dt/2)
        k2y = (x[i] + k1x*dt/2)*(rho - z[i] - k1z*dt/2) - y[i] - k1y*dt/2
        k2z = (x[i] + k1x*dt/2)*(y[i] + k1y*dt/2) - beta*(z[i] + k1z*dt/2)

        k3x = sigma * (y[i] + k2y*dt/2 - x[i] - k2x*dt/2)
        k3y = (x[i] + k2x*dt/2)*(rho - z[i] - k2z*dt/2) - y[i] - k2y*dt/2
        k3z = (x[i] + k2x*dt/2)*(y[i] + k2y*dt/2) - beta*(z[i] + k2z*dt/2)

        k4x = sigma * (y[i] + k3y*dt - x[i] - k3x*dt)
        k4y = (x[i] + k3x*dt)*(rho - z[i] - k3z*dt) - y[i] - k3y*dt
        k4z = (x[i] + k3x*dt)*(y[i] + k3y*dt) - beta*(z[i] + k3z*dt)

        x[i+1] = x[i] + (k1x + 2*k2x + 2*k3x + k4x)*dt/6

        y[i+1] = y[i] + (k1y + 2*k2y + 2*k3y + k4y)*dt/6
        
        z[i+1] = z[i] + (k1z + 2*k2z + 2*k3z + k4z)*dt/6

xdiff = np.zeros(N)
ydiff = np.zeros(N)
zdiff = np.zeros(N)

dist = np.zeros(N)

for i in range(N-1):
    xdiff[i] = norm(x1[i] - x2[i])
    ydiff[i] = norm(y1[i] - y2[i])
    zdiff[i] = norm(z1[i] - z2[i])

    dist[i] = np.sqrt( norm(x1[i] - x2[i])**2 + norm(y1[i] - y2[i])**2 + norm(z1[i] - z2[i])**2 )
    
    t[i+1] = t[i] + dt

# Animating

plt.style.use('dark_background')

fig = plt.figure(figsize=(14,7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)

ax1.xaxis.set_pane_color((0,0,0,1))
ax1.yaxis.set_pane_color((0,0,0,1))
ax1.zaxis.set_pane_color((0,0,0,1))

ax1.grid(False)

xmin = np.min([np.min(x1),np.min(x2)])
xmax = np.max([np.max(x1),np.max(x2)])

ymin = np.min([np.min(y1),np.min(y2)])
ymax = np.max([np.max(y1),np.max(y2)])

zmin = np.min([np.min(z1),np.min(z2)])
zmax = np.max([np.max(z1),np.max(z2)])

distmax = np.max(dist)

ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_zlim(zmin, zmax)

ax2.set_xlim(0, T)
ax2.set_ylim(0, distmax)



def set_xyz(line, x_new, y_new, z_new=None):

    line.set_data(x_new,y_new)

    if z_new is not None:
        line.set_3d_properties(z_new)

from matplotlib.animation import FuncAnimation

(line1,) = ax1.plot([],[],[], color='blue', linewidth=0.5)
(line2,) = ax1.plot([],[],[], color='yellow', linewidth=0.5)
(line3,) = ax2.plot([], [])
(line4,) = ax2.plot([], [], color='red', linewidth=0.5)
(line5,) = ax2.plot([], [], color='green', linewidth=0.5)
(line6,) = ax2.plot([], [], color='blue', linewidth=0.5)

(endpoint1,) = ax1.plot(x1[0], y1[0], z1[0], marker='o', markersize=3, color='blue')
(endpoint2,) = ax1.plot(x2[0], y2[0], z2[0], marker='o', markersize=3, color='yellow')

ax2.set_xlim(0,5)

def update(frame):
    set_xyz(line1, x1[:frame], y1[:frame], z1[:frame])
    set_xyz(line2, x2[:frame], y2[:frame], z2[:frame])
    set_xyz(endpoint1, x1[frame], y1[frame], z1[frame])
    set_xyz(endpoint2, x2[frame], y2[frame], z2[frame])

    set_xyz(line3, t[:frame], dist[:frame])

    if 4 <= t[frame] <= T - 1:
        ax2.set_xlim(t[frame] - 4, t[frame] + 1)

    set_xyz(line4, t[:frame], xdiff[:frame])
    set_xyz(line5, t[:frame], ydiff[:frame])
    set_xyz(line6, t[:frame], zdiff[:frame])
    
    return line1, line2, endpoint1, endpoint2, line3, line4, line5, line6

steps = 10

frames = range(0, N, steps)


ani = FuncAnimation(fig, update, frames=frames, interval=20)


plt.show()


