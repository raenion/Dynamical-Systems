import numpy as np
import matplotlib.pyplot as plt


# we NO LONGER assume sun is fixed, i.e. we compute gravitational pull of Earth on Sun
# due to conservation of angular momentum orbit lies in a plane, say the xy plane




# Turn "earth" --> "planet"

# make planet bigger and closer to sun so both planet's orbit and sun's wobble can be seen easily within same frame of reference

# apply all these changes to "animation wobble" script 

# make markdown of 2-body problem. Blow ups? stability? periodic trajectories? centre of mass (simulate?)?

T = 365*24*3600*4
dt = 3600
N = int(T/dt) + 1

G = 6.6734810e-11 # m^3 / (kg s^2)
m_sun = 1.989e30 # kg
m_earth = 5.972e24 # kg
R = 1.496e11 # m   (average radius of earth's orbit around sun)




r0_earth = [R, 0]
v0_earth = [0, 29.8e3]

r0_sun = [0, 0]
v0_sun = [0, 0]


r_earth = np.zeros((N,2))
v_earth = np.zeros((N,2))

r_sun = np.zeros((N,2))
v_sun = np.zeros((N,2)) 


r_earth[0] = r0_earth
v_earth[0] = v0_earth

r_sun[0] = r0_sun
v_sun[0] = v0_sun      # Can make system momentum zero by counterbalancing planet'S momentum 

norm = np.linalg.norm

def gravitational_acceleration(r1, r2, m2):

    return G*m2*(r2 - r1)/(norm(r2 - r1)**3)

a = gravitational_acceleration

for i in range(N-1):

    # dont call a so much

    r_earth[i+1] = r_earth[i] + v_earth[i] * dt + 1/2*a(r_earth[i], r_sun[i], m_sun) * dt**2    
    r_sun[i+1] = r_sun[i] + v_sun[i] * dt + 1/2*a(r_sun[i], r_earth[i], m_earth) * dt**2


    v_earth[i+1] = v_earth[i] + 1/2*(a(r_earth[i], r_sun[i], m_sun) + a(r_earth[i+1], r_sun[i+1], m_sun) ) * dt
    v_sun[i+1] = v_sun[i] + 1/2*( a(r_sun[i], r_earth[i], m_earth) + a(r_sun[i+1], r_earth[i+1], m_earth) ) * dt

fig, ax = plt.subplots(figsize=(8,8))

ax.set_aspect('equal', 'box')

ax.plot(r_earth[:,0], r_earth[:,1])
ax.plot(r_sun[:,0], r_sun[:,1])

plt.show()


