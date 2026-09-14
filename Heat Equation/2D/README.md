# 2D Heat Equation

Solves
$$
u_t - \nu (u_{xx} + u_{yy}) = 0
$$
on $[-1, 1] \times [-1, 1]$ using explicit Euler (FTCS) in time and centered differences in space.

## Contents

- [Dirichlet](Dirichlet) — fixed boundary values; animated, static colormap, and non-animated variants.
- [Periodic](Periodic) — periodic boundary conditions, animated.

## Notes

- Time step is clamped to the 2D stability bound $dt \le \min(dx^2, dy^2)/4\nu$.
- Initial conditions are swappable via commented-out alternatives at the top of each script.
