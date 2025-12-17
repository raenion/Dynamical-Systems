# Using Verlet algorithm to solve Newton's equations for 1D simple harmonic oscillator

import numpy as np
import matplotlib.pyplot as plt

T = 10
dt = 0.01
N = int(T/dt) + 1

k = 1
m = 1

r = np.zeros(N)
v = np.zeros(N)
t = np.zeros(N)

r0 = 1
v0 = 0

r[0] = r0
v[0] = v0


def acc_SHO(x, k=k, m=m):
    return -k*x/m


for i in range(N-1):

    r[i+1] = r[i] + v[i] * dt + acc_SHO(r[i]) * dt**2

    v[i+1] = v[i] + 1/2 * ( acc_SHO(r[i]) + acc_SHO(r[i+1]) ) * dt

    t[i+1] = t[i] + dt


def analsolx(time, k=k, m=m):
    w = np.sqrt(k/m)
    A = 1
    phi = np.pi/2
    return A*np.sin(w*time + phi)          # how to determine A and phi analytically? 

def analsolv(time, k=k, m=m):
    w = np.sqrt(k/m)
    A = 1
    phi = np.pi/2
    return A*w*np.cos(w*time + phi)



fig = plt.figure(figsize=(14,7))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(t, r, label='numerical solution')
ax1.plot(t, analsolx(t), label='analytical solution')


ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(r, v, label='numerical solution')
ax2.plot(analsolx(t), analsolv(t), label='analytical solution')

# add x and y labels, plot titles

ax1.legend() # move to good position
ax2.legend() # move to good position



plt.show()
