# Burgers' Equation

Solves the 1D viscous Burgers' equation
$$
u_t + u u_x = \mu u_{xx}
$$
on $[-\pi, \pi]$ with Dirichlet boundary conditions, using explicit Euler time-stepping and second-order finite differences.

## Contents

- [1Dburgers.py](1Dburgers.py) — baseline solver.
- [Burgers-Dirchlet.py](Burgers-Dirchlet.py) — higher-resolution run with an animated solution.

## Notes

- Explicit Euler needs a small time step for stability, more so than pure diffusion since advection also contributes; each script asserts $dt \le dx^2/2\mu$ before running.
