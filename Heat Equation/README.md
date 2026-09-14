# Heat Equation

Numerical solutions of the diffusion equation
$$
u_t = \nu (u_{xx} + u_{yy})
$$
using the explicit Euler (FTCS) scheme, in one and two spatial dimensions.

## Contents

- [1D](1D) — Dirichlet and periodic boundary conditions on $[-1, 1]$.
- [2D](2D) — Dirichlet and periodic boundary conditions on $[-1, 1] \times [-1, 1]$.

## Notes

- Time step is checked against the FTCS stability bound ($dt \le dx^2/2\nu$ in 1D) and adjusted automatically if violated.
- Several scripts store the full space–time solution to support post-processing and animation.
