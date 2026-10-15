# cascade

Learning-oriented turbulence simulations: mixing-length RANS in channel/pipe flow, 2D and
3D pseudo-spectral DNS (dealiased Fourier, integrating-factor RK4), with spectral and
structure-function diagnostics. Python first; C++/FFTW ports of validated code later.

## Purpose

A physicist's workbench for studying turbulence by building the numerics oneself, following
Guyon, Hulin & Petit, *Hydrodynamique physique* (ch. 12) and Pope, *Turbulent Flows*.

The numerical core (wavenumber grid and dealiasing, nonlinear terms, Leray projection, time
integrators, forcing, initial conditions, RANS solvers) is written by hand by the owner. Those
functions are marked `# CORE`; until implemented they are stubs whose docstrings state the
governing equation and the reference section, and whose tests are skipped. Diagnostics, I/O,
plotting, tests and CI live around them. See [`CLAUDE.md`](CLAUDE.md).

## Roadmap

**A. Mixing-length RANS (channel flow).** Mean momentum balance with a van Driest–damped
mixing length capped in the outer layer; recover the viscous sublayer `u⁺ = y⁺` and the log
law `u⁺ = κ⁻¹ ln y⁺ + B` (Pope §7.1, §10.3). → `cascade.rans.channel`

**B. 2D forced turbulence.** Vorticity–streamfunction pseudo-spectral solver, white-in-time
ring forcing at `k_f`, linear drag plus hyperviscosity. Kraichnan's double cascade: inverse
energy cascade `E(k) ∝ ε^{2/3} k^{-5/3}` for `k < k_f`, direct enstrophy cascade
`E(k) ∝ η^{2/3} k^{-3}` for `k > k_f`, with the corresponding constant fluxes.
→ `cascade.spectral2d`

**C. 3D forced homogeneous isotropic turbulence.** Rotational-form Navier–Stokes with Leray
projection, constant-power forcing. K41: `E(k) = C ε^{2/3} k^{-5/3}` in the inertial range,
the 4/5 law `S₃(r) = -(4/5) ε r`, and intermittency through the anomalous scaling exponents
`ζ_p` of higher-order structure functions (Pope ch. 6). → `cascade.spectral3d`

Diagnostics shared by B and C: shell spectra (done), transfer and flux, dissipation and
characteristic scales, structure functions, PDFs. → `cascade.diagnostics`

## Layout

```
src/cascade/
  grid.py                 CORE  wavenumbers, k², 2/3 mask (rfft layout)
  rans/channel.py         CORE  mixing-length channel solver
  spectral2d/solver.py    CORE  2D RHS, RK4-IF, CFL, forcing, drag + hyperviscosity
  spectral3d/solver.py    CORE  3D RHS, Leray projection, RK4-IF, forcing, initial condition
  diagnostics/spectra.py        shell-summed E(k) in 2D/3D (independent of grid.py)
  diagnostics/{flux,structure,stats}.py, io.py, plotting.py, cli.py   (placeholders)
tests/                          pytest; CORE tests skip until the CORE exists
```

## Conventions

Periodic box `[0, 2π)^d`, integer wavenumbers, `scipy.fft.rfftn(..., norm="forward")` so that
the coefficients are true Fourier coefficients and `Σ_k |û_k|² = ⟨u²⟩` (weight 2 on stored
modes with `0 < k_last < N/2`). float64 throughout.

## Running the tests

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```sh
uv sync                     # create .venv with runtime + dev dependencies
uv run pytest               # tests; CORE tests are reported as skipped until implemented
uv run ruff check           # lint
uv run ruff format --check  # formatting
```

`uv run pytest -ra` lists each skipped test with the CORE function it is waiting for.
Optional FFTW backend: `uv sync --extra fftw` (installs `pyfftw`).
