"""3D rotational-form Navier–Stokes solver (CORE: skipped until implemented)."""

import numpy as np
import pytest
from scipy import fft

from cascade import grid
from cascade.diagnostics.spectra import half_spectrum_weights
from cascade.spectral3d import solver

N = 16
SHAPE = (N, N, N)
AXES = (1, 2, 3)


def _wsum(a):
    """Full-spectrum sum from rfft-half values (Hermitian-symmetric quantity)."""
    return float(np.sum(half_spectrum_weights(SHAPE) * a))


def _relative_divergence(v_hat, k, k2):
    r"""‖k·v̂‖ / ‖|k| v̂‖ (full-spectrum ℓ² norms): dimensionless, 0 for solenoidal v."""
    div = sum(ki * vi for ki, vi in zip(k, v_hat, strict=True))
    num = _wsum(np.abs(div) ** 2)
    den = _wsum(k2 * np.sum(np.abs(v_hat) ** 2, axis=0))
    return np.sqrt(num / den)


@pytest.fixture
def ops(requires_core):
    k = grid.wavenumbers(SHAPE)
    k2 = grid.k_squared(k)
    mask = grid.dealias_mask(SHAPE)
    return k, k2, mask


@pytest.fixture
def random_hat(rng):
    """Coefficients of a white-noise vector field: not solenoidal, all modes populated."""
    return fft.rfftn(rng.standard_normal((3, *SHAPE)), axes=AXES, norm="forward")


@pytest.fixture
def u_hat(ops, random_hat):
    """Dealiased, solenoidal random velocity built only from the projection."""
    k, k2, mask = ops
    return solver.leray_project(random_hat * mask, k, k2)


def test_leray_projection_is_divergence_free(ops, random_hat):
    k, k2, _ = ops
    assert _relative_divergence(random_hat, k, k2) > 0.1  # the input is far from solenoidal
    v = solver.leray_project(random_hat, k, k2)
    assert _relative_divergence(v, k, k2) <= 1e-12


def test_leray_projection_is_idempotent_and_keeps_mean(ops, random_hat):
    k, k2, _ = ops
    v = solver.leray_project(random_hat, k, k2)
    np.testing.assert_allclose(solver.leray_project(v, k, k2), v, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(v[:, 0, 0, 0], random_hat[:, 0, 0, 0])


def test_nonlinear_term_is_divergence_free(ops, u_hat):
    k, k2, mask = ops
    nl = solver.nonlinear_rhs(u_hat, k, k2, mask)
    assert np.all(nl[:, ~mask] == 0)
    assert _relative_divergence(nl, k, k2) <= 1e-12


def test_nonlinear_term_conserves_energy(ops, u_hat):
    """Σ Re(û*·N̂) = 0 to round-off: u·(u×ω) = 0 and the truncation is exact (2/3 rule)."""
    k, k2, mask = ops
    nl = solver.nonlinear_rhs(u_hat, k, k2, mask)
    transfer = _wsum(np.sum(np.real(np.conj(u_hat) * nl), axis=0))
    scale = _wsum(np.sqrt(np.sum(np.abs(u_hat) ** 2, axis=0) * np.sum(np.abs(nl) ** 2, axis=0)))
    assert scale > 0
    assert abs(transfer) <= 1e-12 * scale


def test_random_initial_condition(ops, rng):
    k, k2, mask = ops
    energy = 0.7
    u = solver.random_solenoidal_field(SHAPE, k_peak=3.0, energy=energy, rng=rng)
    assert u.shape == (3, N, N, N // 2 + 1)
    assert _relative_divergence(u, k, k2) <= 1e-12
    assert 0.5 * _wsum(np.sum(np.abs(u) ** 2, axis=0)) == pytest.approx(energy, rel=1e-12)
    assert np.all(u[:, ~mask] == 0)
    np.testing.assert_array_equal(u[:, 0, 0, 0], 0)
    # Hermitian-consistent: the coefficients describe a real field.
    roundtrip = fft.rfftn(
        fft.irfftn(u, s=SHAPE, axes=AXES, norm="forward"), axes=AXES, norm="forward"
    )
    np.testing.assert_allclose(roundtrip, u, rtol=0, atol=1e-14 * np.max(np.abs(u)))


def test_constant_power_forcing_injects_epsilon(ops, u_hat):
    """Σ Re(û*·f̂) = ε exactly, forcing confined to 0 < |k| <= k_f and solenoidal."""
    k, k2, _ = ops
    epsilon, k_f = 0.1, 2.5
    f = solver.constant_power_forcing(u_hat, k2, k_f, epsilon)
    assert _wsum(np.sum(np.real(np.conj(u_hat) * f), axis=0)) == pytest.approx(epsilon, rel=1e-12)
    outside = (k2 > k_f**2) | (k2 == 0)
    assert np.all(f[:, outside] == 0)
    assert _relative_divergence(f, k, k2) <= 1e-12
