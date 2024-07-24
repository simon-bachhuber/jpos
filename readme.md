# JPOS: Joint Position Estimation

If `jpos.solve(..., order=0)`, then this algorithm reduces to https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6402423.

## Installation
`pip install git+https://github.com/SimiPixel/jpos.git`

## Usage
```python
import numpy as np
from jpos import solve

# sampling rate
hz: float = 60

# number of timesteps
N : int   = 1000

# acc1, acc2: Nx3 in m/s**2
acc1 = np.zeros((N, 3))
acc2 = np.zeros((N, 3))

# gyr1, gyr2: Nx3 in rad/s
gyr1 = np.zeros((N, 3))
gyr2 = np.zeros((N, 3))

# phi is the hinge joint angle over time in radians: Nx1
# phi can also be set to `None` if `order`=0, then a constant offset vector r is estimated
# instead of a function r(phi)
phi = np.zeros((N, 1))

# solve for the vector that connects the hinge joint's center of rotation to the
# - r1: center of the first imu, in coordinates of the imu1
# - r2: center of the second imu, in coordinates of the imu2
r1, r2, infos = solve(acc1, gyr1, acc2, gyr2, phi, hz, order=2)

# the connection to the first imu is assumed to be rigid and not depend on the
# hinge joint angle phi, r1 is given in meters
assert r1.shape == (3,)

# the connection to the second imu is assumed to be a time-varying function
# that is modeled as a polynomial of order `order` of the hinge joint angle phi
# the returned result is then this estimated function r2(phi) evaluated at phi(t)
# which gives the connection vector at every timestep, r2 is given in meters
assert r2.shape == (N, 3)

# you can also access the estimated r2 vector as a function of phi using
assert infos["r2_grid_m"] == (50, 3)
# which by default uses 50 grid points, located at 
assert infos["phi_grid_rad"] == (50,)
# which are distributed uniformly in [min(phi), max(phi)]
```