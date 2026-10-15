"""2D vorticity–streamfunction solver (CORE: skipped until implemented)."""

import numpy as np
import pytest
from scipy import fft

from cascade import grid
from cascade.diagnostics.spectra import half_spectrum_weights
from cascade.spectral2d import solver

N = 32
SHAPE = (N, N)


def _xy():
    x = 2 * np.pi * np.arange(N) / N
    return np.meshgrid(x, x, indexing="ij")


def _rfft(f):
    return fft.rfft2(f, norm="forward")


def _irfft(f_hat):
    return fft.irfft2(f_hat, s=SHAPE, norm="forward")


def _wsum(a):
    """Full-spectrum sum from rfft-half values (Hermitian-symmetric quantity)."""
    return float(np.sum(half_spectrum_weights(SHAPE) * a))


def _band_limited_vorticity(rng, k_max=4, peak=2.0):
    """Random zero-mean trigonometric polynomial, |m|, |n| <= k_max, max|ω| = peak."""
    x, y = _xy()
    w = np.zeros(SHAPE)
    for m in range(-k_max, k_max + 1):
        for n in range(k_max + 1):
            if (m, n) != (0, 0):
                a, b = rng.standard_normal(2) / np.hypot(m, n)
                w += a * np.cos(m * x + n * y) + b * np.sin(m * x + n * y)
    return peak * w / np.max(np.abs(w))


@pytest.fixture
def ops(requires_core):
    kx, ky = grid.wavenumbers(SHAPE)
    k2 = grid.k_squared((kx, ky))
    mask = grid.dealias_mask(SHAPE)
    return kx, ky, k2, mask


def _psi_hat(w_hat, k2):
    return np.divide(w_hat, k2, out=np.zeros_like(w_hat), where=k2 > 0)


def _energy(w_hat, k2):
    return 0.5 * _wsum(
        np.abs(w_hat) ** 2 * np.divide(1.0, k2, out=np.zeros(k2.shape), where=k2 > 0)
    )


def _enstrophy(w_hat):
    return 0.5 * _wsum(np.abs(w_hat) ** 2)


@pytest.mark.parametrize(
    ("order", "nu", "mu"),
    [(1, 0.05, 0.0), (1, 0.05, 0.1), (4, 0.01, 0.0), (2, 0.02, 0.05)],
)
def test_taylor_green_exact_decay(ops, order, nu, mu):
    """ψ = sin x sin y, ω = 2ψ: J(ψ, ω) = 0, so ω(t) = ω(0) exp[-(ν 2^p + μ) t] exactly."""
    kx, ky, k2, mask = ops
    x, y = _xy()
    omega0 = 2 * np.sin(x) * np.sin(y)
    lin = solver.linear_operator(k2, nu, mu, order)

    w_hat = _rfft(omega0)
    dt, n_steps = 0.05, 40
    for _ in range(n_steps):
        w_hat = solver.rk4_if_step(
            w_hat, dt, lin, lambda w: solver.nonlinear_rhs(w, kx, ky, k2, mask)
        )

    exact = omega0 * np.exp(-(nu * 2.0**order + mu) * dt * n_steps)
    np.testing.assert_allclose(_irfft(w_hat), exact, rtol=0, atol=1e-10)


def test_cfl_taylor_green(ops):
    """TG velocity u = sin x cos y, v = -cos x sin y: max(|u| + |v|) = max|sin(x ± y)| = 1."""
    kx, ky, k2, _ = ops
    x, y = _xy()
    w_hat = _rfft(2 * np.sin(x) * np.sin(y))
    dt = solver.cfl_time_step(w_hat, kx, ky, k2, cfl=0.4)
    assert dt == pytest.approx(0.4 * (2 * np.pi / N), rel=1e-12)
    assert solver.cfl_time_step(np.zeros_like(w_hat), kx, ky, k2, dt_max=0.1) == 0.1


def test_nonlinear_term_conserves_energy_and_enstrophy(ops, rng):
    """Galerkin-truncated Euler: Σ ψ̂* N̂ = Σ ω̂* N̂ = 0 to round-off (instantaneous)."""
    kx, ky, k2, mask = ops
    w_hat = _rfft(_band_limited_vorticity(rng))
    nl = solver.nonlinear_rhs(w_hat, kx, ky, k2, mask)
    assert np.all(nl[~mask] == 0)
    psi_hat = _psi_hat(w_hat, k2)

    d_energy = _wsum(np.real(np.conj(psi_hat) * nl))
    d_enstrophy = _wsum(np.real(np.conj(w_hat) * nl))
    assert abs(d_energy) <= 1e-12 * _wsum(np.abs(psi_hat) * np.abs(nl))
    assert abs(d_enstrophy) <= 1e-12 * _wsum(np.abs(w_hat) * np.abs(nl))
    assert _wsum(np.abs(nl)) > 0  # the test field is genuinely nonlinear


def test_inviscid_time_integration_conserves_invariants(ops, rng):
    """ν = μ = 0, no forcing: E and Z drift only by the RK4 truncation error over ~1 turnover."""
    kx, ky, k2, mask = ops
    lin = solver.linear_operator(k2, 0.0, 0.0)
    w_hat = _rfft(_band_limited_vorticity(rng))
    e0, z0 = _energy(w_hat, k2), _enstrophy(w_hat)

    dt, n_steps = 0.005, 200
    for _ in range(n_steps):
        w_hat = solver.rk4_if_step(
            w_hat, dt, lin, lambda w: solver.nonlinear_rhs(w, kx, ky, k2, mask)
        )

    assert _energy(w_hat, k2) == pytest.approx(e0, rel=1e-6)
    assert _enstrophy(w_hat) == pytest.approx(z0, rel=1e-6)
