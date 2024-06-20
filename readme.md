# JPOS: Joint Position Estimation

## Installation
`pip install git+https://github.com/SimiPixel/jpos.git`

## Usage
```python
from jpos import solve

hz: float = 60

# phi is the hinge joint angle over time in radians: Nx1
# acc1, acc2: Nx3 in m/s**2
# gyr, gyr2: Nx3 in rad/s
r1, r2, infos = solve(acc1, gyr1, acc2, gyr2, phi, hz)
# joint to imu1 vector in meters: r1.shape = 3
# timeseries of joint to imu2 vector in meters: r2.shape = Nx3
# both joint to imu1/2 vectors are given in coordinates of the local
# sensor coordinate system
```