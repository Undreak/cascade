"""Mixing-length channel flow (CORE: skipped until implemented)."""

import numpy as np
import pytest

from cascade.rans.channel import solve_channel

RE_TAU = 1000.0
N_POINTS = 2001


@pytest.fixture
def profile(requires_core):
    return solve_channel(RE_TAU, N_POINTS)


def test_output_contract(profile):
    y, u = profile
    assert y.shape == u.shape == (N_POINTS,)
    assert y[0] == 0.0
    assert y[-1] == pytest.approx(RE_TAU, rel=1e-12)
    assert u[0] == 0.0
    assert np.all(np.diff(y) > 0)
    assert np.all(np.diff(u) > 0)


def test_viscous_sublayer(profile):
    """u⁺ = y⁺ for y⁺ <= 2: ν_t⁺ ~ (κ y⁺²/A⁺)² < 1e-2 there."""
    y, u = profile
    near = (y > 0) & (y <= 2.0)
    assert np.count_nonzero(near) >= 2
    np.testing.assert_allclose(u[near], y[near], rtol=1e-2)


def test_log_law(profile):
    """Fit u⁺ = κ⁻¹ ln y⁺ + B on 50 <= y⁺ <= 150 (y/δ <= 0.15) at Re_τ = 1000.

    Tolerances are not tight because the model itself departs from a pure log law in
    this window: van Driest damping still steepens the profile by a few % at y⁺ = 50,
    and the total stress 1 - y/δ lowers it by up to ~8 % at y⁺ = 150 (local slope
    ∝ √(1 - y/δ) / (1 - e^{-y⁺/A⁺})). These bounds are an a-priori estimate, not yet checked
    against an implementation: revisit them once solve_channel exists.
    """
    y, u = profile
    y_fit = np.geomspace(50.0, 150.0, 41)
    slope, intercept = np.polyfit(np.log(y_fit), np.interp(y_fit, y, u), 1)
    kappa_fit = 1.0 / slope
    assert kappa_fit == pytest.approx(0.41, rel=0.10)
    assert intercept == pytest.approx(5.2, abs=1.0)
