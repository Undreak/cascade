"""RK4 integrating-factor steps, 2D and 3D (CORE: skipped until implemented).

The integrator is tested in isolation on the Bernoulli equation

    du/dt = λ u + u²,

whose exact solution follows from z = 1/u, dz/dt = -λ z - 1:

    u(t) = 1 / [ (1/u0 + 1/λ) e^{-λ t} - 1/λ ].
"""

import numpy as np
import pytest

from cascade.spectral2d import solver as solver2d
from cascade.spectral3d import solver as solver3d

SOLVERS = [
    pytest.param(solver2d, id="2d"),
    pytest.param(solver3d, id="3d"),
]

LAM = np.array([-0.5, -1.0, -2.0])
U0 = np.full(3, 0.3 + 0.0j)


def _exact(t):
    return 1.0 / ((1.0 / U0 + 1.0 / LAM) * np.exp(-LAM * t) - 1.0 / LAM)


def _integrate(module, n_steps, t_end=1.0):
    dt = t_end / n_steps
    u = U0.copy()
    for _ in range(n_steps):
        u = module.rk4_if_step(u, dt, LAM, lambda v: v * v)
    return u


@pytest.mark.parametrize("module", SOLVERS)
def test_rk4_temporal_order(requires_core, module):
    """Halving Δt divides the global error by 2⁴ = 16."""
    err = [np.max(np.abs(_integrate(module, n) - _exact(1.0))) for n in (40, 80)]
    assert err[1] > 1e-14  # still above round-off: the ratio is meaningful
    assert err[0] / err[1] == pytest.approx(16.0, rel=0.15)


@pytest.mark.parametrize("module", SOLVERS)
def test_linear_part_is_exact(requires_core, module):
    """With N = 0 the integrating factor gives e^{LΔt} for any Δt (no stability limit)."""
    lin = np.array([0.0, -1.0, -1e4])
    u0 = np.array([1.0 + 2.0j, 1.0, 1.0])
    u1 = module.rk4_if_step(u0, 0.5, lin, np.zeros_like)
    np.testing.assert_allclose(u1, u0 * np.exp(0.5 * lin), rtol=1e-14, atol=0)


@pytest.mark.parametrize("module", SOLVERS)
def test_step_does_not_modify_input(requires_core, module):
    u0 = U0.copy()
    module.rk4_if_step(u0, 0.1, LAM, lambda v: v * v)
    np.testing.assert_array_equal(u0, U0)
