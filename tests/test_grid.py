"""Wavenumber grid and 2/3 mask (CORE: skipped until implemented)."""

import numpy as np
import pytest
from scipy import fft

from cascade import grid

# Non-square shapes catch transposed axes.
SHAPES = [(16, 16), (12, 16), (16, 16, 16), (8, 12, 16)]


def _coords(shape):
    axes = [2 * np.pi * np.arange(n) / n for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def _rfft_shape(shape):
    return (*shape[:-1], shape[-1] // 2 + 1)


@pytest.mark.parametrize("shape", SHAPES)
def test_wavenumbers_broadcast_to_rfft_layout(requires_core, shape):
    k = grid.wavenumbers(shape)
    assert len(k) == len(shape)
    for ki in k:
        assert ki.dtype == np.float64
        assert np.broadcast_shapes(ki.shape, _rfft_shape(shape)) == _rfft_shape(shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_spectral_derivative_is_exact(requires_core, shape):
    """∂_i of a band-limited trigonometric polynomial: i k_i û, exact to round-off."""
    d = len(shape)
    x = _coords(shape)
    m = (3, -2, 1)[:d]
    phase = sum(mi * xi for mi, xi in zip(m, x, strict=True))
    f = np.sin(phase) + 0.5 * np.cos(2 * x[-1])
    f_hat = fft.rfftn(f, norm="forward")
    k = grid.wavenumbers(shape)
    for axis in range(d):
        df = fft.irfftn(1j * k[axis] * f_hat, s=shape, norm="forward")
        exact = m[axis] * np.cos(phase) - (axis == d - 1) * np.sin(2 * x[-1])
        np.testing.assert_allclose(df, exact, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_k_squared_is_minus_laplacian_symbol(requires_core, shape):
    d = len(shape)
    x = _coords(shape)
    m = (3, -2, 1)[:d]
    f = np.cos(sum(mi * xi for mi, xi in zip(m, x, strict=True)))
    k2 = grid.k_squared(grid.wavenumbers(shape))
    assert k2.shape == _rfft_shape(shape)
    lap = fft.irfftn(-k2 * fft.rfftn(f, norm="forward"), s=shape, norm="forward")
    np.testing.assert_allclose(lap, -sum(mi**2 for mi in m) * f, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_dealias_mask_is_two_thirds_box(requires_core, shape):
    mask = grid.dealias_mask(shape)
    assert mask.dtype == np.bool_
    assert mask.shape == _rfft_shape(shape)
    k = grid.wavenumbers(shape)
    keep = np.ones(_rfft_shape(shape), dtype=bool)
    for ki, n in zip(k, shape, strict=True):
        keep &= np.abs(ki) < n / 3
    np.testing.assert_array_equal(mask, keep)
