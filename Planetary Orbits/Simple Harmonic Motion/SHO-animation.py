import numpy as np 
import matplotlib.pyplot as plt
import math

T = 35
dt = 0.01
N = int(T/dt) + 1
k = 1
m = 1
w = np.sqrt(k/m)
period = 2*np.pi/w

x0 = 1
v0 = 0

tvals = np.linspace(0, T, N)

A = np.sqrt(x0**2 + v0**2/(w**2))
phi = np.arctan(-v0/(x0*w))


def x(time):
    return A*np.cos(w*time + phi)

def v(time):
    return -A*w*np.sin(w*time + phi)


xvals = x(tvals)
vvals = v(tvals)

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(12,8))

(line,) = ax.plot([], [], zorder = 2)
(endpoint,) = ax.plot([], [], marker='o', markersize=10, color='red', zorder = 3)
(oscillation_domain,) = ax.plot([], [], color='black', linewidth=0.5, zorder = 1)

ax.set_xlim(-1.5,1.5)

# Delayed drawing parameters:

time_until_draw = 2*period
frame_until_draw = math.ceil(time_until_draw/dt)
time_window = period

xdomain = np.linspace(-A, A, int(2*A*100)) # Add ticks to make it look like an axis?


step = 5

frames = range(0, N, step)

def update(frame):

    

    if frame == 0:
        ax.set_ylim(-period, period)


    if tvals[frame] < time_until_draw:
        oscillation_domain.set_data(xdomain, [0])
        endpoint.set_data([xvals[frame]], [0])


    
    else:
        oscillation_domain.set_data(xdomain, [tvals[frame]])
        line.set_data(xvals[frame_until_draw:frame], tvals[frame_until_draw:frame])
        endpoint.set_data([xvals[frame]], [tvals[frame]])

        if tvals[frame] > T - time_window:
            pass

        else:
             ax.set_ylim(tvals[frame] - time_window, tvals[frame] + time_window)


    
    return oscillation_domain, line, endpoint



ani = FuncAnimation(fig, update, frames=frames, interval=20, repeat=False)

plt.show()
