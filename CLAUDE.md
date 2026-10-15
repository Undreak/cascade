# CLAUDE.md — cascade

## Purpose
Learning project: turbulence simulations to study the physics (Guyon, Hulin & Petit,
*Hydrodynamique physique*, ch. 12; Pope, *Turbulent Flows*). The owner is a physicist
who implements the numerical core himself to learn it.

## Hard rule: human-written CORE
Functions marked `# CORE` are written by the owner. Never implement them — not to make
a test pass, not as an "example", not when a task indirectly requires them. You may:
- write their signatures and docstrings (governing equation in LaTeX + reference section),
  with body `raise NotImplementedError`;
- write tests against them;
- review them and explain bugs, without writing the fix unless explicitly asked.

CORE = wavenumber grid & dealiasing mask, nonlinear term / RHS, Leray projection,
time integrators, forcing, initial conditions, mixing-length / RANS solvers.

If a task seems to require CORE code, stop and say so in the PR description.

## Your scope
Diagnostics (shell spectra, transfer/flux, dissipation, characteristic scales,
structure functions, PDFs), I/O (HDF5), plotting, CLI/config, tests, CI, benchmarks,
C++ ports of *validated* Python code.

## Conventions
- Python >= 3.12, uv, src/ layout, ruff (lint + format), pytest, type hints,
  NumPy-style docstrings.
- Periodic domain [0, 2π)^d, integer wavenumbers, `scipy.fft` real transforms
  (rfftn/irfftn) with `norm="forward"`: û are true Fourier coefficients,
  so Σ_k |û|² = <u²> (with ×2 weight on half-spectrum modes 0 < k_last < N/2).
- float64 by default.
- Tests on CORE functions use the `requires_core` fixture: skip on NotImplementedError.
- Analytical/exact checks first (Taylor–Green, Parseval, ∇·u = 0, Euler invariants).
- One topic per PR. A diagnostics PR never touches CORE files.
- Explain non-obvious numerical choices (normalization, weights, binning) in the PR.
