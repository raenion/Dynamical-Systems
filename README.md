# Dynamical Systems

A collection of numerical simulations of classical dynamical systems — PDEs solved by finite differences and ODEs solved by direct integration, each with visualization or animation.

## Contents

- [Heat Equation](Heat%20Equation) — 1D and 2D diffusion, Dirichlet and periodic boundary conditions.
- [Wave Equation](Wave%20Equation) — 1D and 2D wave propagation via central differences.
- [Burgers Equation](Burgers%20Equation) — 1D viscous Burgers' equation (nonlinear advection-diffusion).
- [Attractors](Attractors) — Lorenz system, integrated with RK4, including chaos and integrator-comparison studies.
- [Planetary Orbits](Planetary%20Orbits) — N-body gravitational dynamics (Earth–Sun, two-body wobble, three-body) and the simple harmonic oscillator.

## Approach

Scripts favor clarity over performance: explicit time-stepping schemes, full space–time solutions stored where useful for analysis, and stability bounds checked (and auto-corrected) at runtime.
