# JPOS: Joint Position Estimation

## Installation
`pip install imt-qpos`

## Usage
```python
from qpos import solve

hz: float = 60

# phi is the hinge joint angle over time in radians: Nx1
# acc1, acc2: Nx3 in m/s**2
# gyr, gyr2: Nx3 in rad/s
r1, r2, infos = solve(acc1, gyr1, acc2, gyr2, phi, hz)
# joint to imu1 vector in meters: r1.shape = (3,)
# joint to imu2 vector in meters evaluated at 50 grid points within [phi_min, phi_max]
# r2.shape = (50, 3)

```