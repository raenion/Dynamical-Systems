# Wave Equation

Numerical solutions of the wave equation
$$
u_{tt} = c^2 \nabla^2 u
$$
using explicit central differences in both time and space, with fixed (Dirichlet) endpoints/boundaries.

## Contents

- [1D](1D) — `1DWave.py` (baseline) and `Ani1Dwave.py` (animated, full solution stored for analysis).
- [2D](2D) — `Ani2Dwave.py`, animated propagation on a rectangular membrane.

## Notes

- The scheme is a leapfrog-style update: each new time slice depends on the two previous ones, so the first step needs a separate initialization (e.g. zero initial velocity).
