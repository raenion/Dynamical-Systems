# Planetary Orbits

N-body gravitational dynamics under Newton's law of gravitation,
$$
\mathbf{F} = -\frac{G m_1 m_2}{r^2}\hat{\mathbf{r}},
$$
integrated numerically and animated.

## Contents

- [Earth-Sun.py](Earth-Sun.py) — two-body orbit with the Sun fixed at the origin (Velocity Verlet); Earth's mass is treated as negligible.
- [Sun-Wobble.py](Sun-Wobble.py) — two-body orbit without the fixed-Sun assumption, so the star wobbles about the center of mass.
- [animation wobble.py](animation%20wobble.py) — animated version of the two-body wobble.
- [3-Body Problem on plane.py](3-Body%20Problem%20on%20plane.py) — three equal-mass bodies interacting under mutual gravity, confined to a plane.
- [Simple Harmonic Motion](Simple%20Harmonic%20Motion) — the simple harmonic oscillator as the simplest continuous dynamical system.

## Notes

- All orbits are planar, a consequence of conservation of angular momentum for the two- and (fixed-plane) three-body cases simulated here.
- Units are SI (meters, seconds, kilograms); the astronomical unit and standard solar/Earth masses are used as reference scales.
