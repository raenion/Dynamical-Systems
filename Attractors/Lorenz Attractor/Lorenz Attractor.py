import numpy as np
import matplotlib.pyplot as plt

T = 35
dt = 0.001
N = int(T/dt) + 1

sigma = 10
rho = 28
beta = 8/3

xe = np.zeros(N)
ye = np.zeros(N)
ze = np.zeros(N)

x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

t = np.zeros(N)

xe[0] = 0
ye[0] = 2
ze[0] = 20


x[0] = 0
y[0] = 2
z[0] = 20


fig = plt.figure(figsize=(14,7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.grid(False)
ax2.grid(False)

for i in range(N-1):
    xe[i+1] = xe[i] + sigma*(ye[i] - xe[i]) * dt
    ye[i+1] = ye[i] + ( xe[i]*(rho - ze[i]) - ye[i] ) * dt
    ze[i+1] = ze[i] + ( xe[i]*ye[i] - beta*ze[i] ) * dt



'''

for i in range(N-1):

    k1x = sigma * (y[i] - x[i])
    k2x = sigma * (y[i] - x[i] - k1x*dt/2)
    k3x = sigma * (y[i] - x[i] - k2x*dt/2)
    k4x = sigma * (y[i] - x[i] - k3x*dt)

    x[i+1] = x[i] + (k1x + 2*k2x + 2*k3x + k4x)*dt/6

    k1y = x[i]*(rho - z[i]) - y[i]
    k2y = x[i]*(rho - z[i]) - y[i] - k1y*dt/2
    k3y = x[i]*(rho - z[i]) - y[i] - k2y*dt/2
    k4y = x[i]*(rho - z[i]) - y[i] - k3y*dt

    y[i+1] = y[i] + (k1y + 2*k2y + 2*k3y + k4y)*dt/6

    k1z = x[i]*y[i] - beta*z[i]
    k2z = x[i]*y[i] - beta*(z[i] + k1z*dt/2)
    k3z = x[i]*y[i] - beta*(z[i] + k2z*dt/2)
    k4z = x[i]*y[i] - beta*(z[i] + k3z*dt)
    
    z[i+1] = z[i] + (k1z + 2*k2z + 2*k3z + k4z)*dt/6

    t[i+1] = t[i] + dt

'''


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

    t[i+1] = t[i] + dt



ax1.plot3D(xe,ye,ze, color='black')
ax2.plot3D(x,y,z, color='black', linewidth=0.1)





plt.show()






