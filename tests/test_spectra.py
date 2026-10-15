"""Shell spectra: Parseval, single-mode placement, mode counting (non-CORE, runs now)."""

import numpy as np
import pytest
from scipy import fft

from cascade.diagnostics.spectra import (
    energy_spectrum,
    half_spectrum_weights,
    shell_mode_count,
    shell_spectrum,
)

# Even and odd sizes (Nyquist plane present / absent), square and non-square grids.
GRID_SHAPES = [(32, 32), (31, 33), (24, 17), (16, 16, 16), (15, 16, 17), (8, 12, 10)]


def _coords(grid_shape):
    axes = [2 * np.pi * np.arange(n) / n for n in grid_shape]
    return np.meshgrid(*axes, indexing="ij")


def _rfft(u, d):
    return fft.rfftn(u, axes=tuple(range(-d, 0)), norm="forward")


@pytest.mark.parametrize("grid_shape", GRID_SHAPES)
@pytest.mark.parametrize("n_components", [None, 1, 3])
def test_parseval_random_field(rng, grid_shape, n_components):
    d = len(grid_shape)
    lead = () if n_components is None else (n_components,)
    u = rng.standard_normal(lead + grid_shape) + 0.3  # nonzero mean exercises shell 0
    k, e = energy_spectrum(_rfft(u, d), grid_shape)

    expected = 0.5 * np.mean(np.sum(u.reshape((-1, *grid_shape)) ** 2, axis=0))
    assert e.dtype == np.float64
    assert k.shape == e.shape
    np.testing.assert_array_equal(k, np.arange(k.size))
    assert np.sum(e) == pytest.approx(expected, rel=1e-12)
    assert np.all(e >= 0)


@pytest.mark.parametrize("grid_shape", GRID_SHAPES)
def test_mode_count_sums_to_grid_size(grid_shape):
    counts = shell_mode_count(grid_shape)
    assert np.sum(counts) == np.prod(grid_shape)
    assert counts[0] == 1


@pytest.mark.parametrize(
    ("grid_shape", "expected"),
    [
        # Shell 1 = {k : 1/4 <= |k|^2 < 9/4} = {|k|^2 = 1, 2}.
        ((32, 32), [1, 8]),  # 4 axis + 4 diagonal lattice points
        ((16, 16, 16), [1, 18]),  # 6 axis + 12 face-diagonal lattice points
    ],
)
def test_mode_count_low_shells(grid_shape, expected):
    np.testing.assert_array_equal(shell_mode_count(grid_shape)[:2], expected)


def test_half_spectrum_weights_parity():
    np.testing.assert_array_equal(half_spectrum_weights((8, 8)).ravel(), [1, 2, 2, 2, 1])
    np.testing.assert_array_equal(half_spectrum_weights((8, 7)).ravel(), [1, 2, 2, 2])
    assert half_spectrum_weights((4, 6, 8)).shape == (1, 1, 5)


@pytest.mark.parametrize(
    ("grid_shape", "mode", "shell"),
    [
        ((32, 32), (3, 4), 5),  # |k| = 5 exactly
        ((32, 32), (-3, 4), 5),  # negative wavenumber on a full axis
        ((32, 32), (5, 0), 5),  # k_last = 0 plane (weight 1)
        ((32, 32), (1, 1), 1),  # |k| = 1.414 -> shell 1
        ((32, 32), (1, 2), 2),  # |k| = 2.236 -> shell 2
        ((32, 32), (2, 2), 3),  # |k| = 2.828 -> shell 3
        ((31, 33), (7, 9), 11),  # |k| = 11.40, odd sizes
        ((16, 16, 16), (1, 2, 2), 3),
        ((16, 16, 16), (0, 3, 0), 3),
        ((15, 16, 17), (-2, 3, 6), 7),
    ],
)
def test_single_mode_lands_in_its_shell(grid_shape, mode, shell):
    d = len(grid_shape)
    amplitude = 1.7
    phase = sum(m * x for m, x in zip(mode, _coords(grid_shape), strict=True))
    u = amplitude * np.cos(phase + 0.4)
    _, e = energy_spectrum(_rfft(u, d), grid_shape)

    # 1/2 <A^2 cos^2> = A^2 / 4, all of it in one shell.
    assert e[shell] == pytest.approx(amplitude**2 / 4, rel=1e-12)
    others = np.delete(e, shell)
    assert np.max(np.abs(others)) < 1e-26


@pytest.mark.parametrize("grid_shape", [(16, 16), (12, 16), (8, 8, 8)])
@pytest.mark.parametrize("axis", [0, -1])
def test_nyquist_mode(grid_shape, axis):
    """cos(N x / 2) = (-1)^j on the grid: energy 1/2 (not 1/4), in shell N/2."""
    d = len(grid_shape)
    n = grid_shape[axis]
    u = np.cos(n // 2 * _coords(grid_shape)[axis])
    _, e = energy_spectrum(_rfft(u, d), grid_shape)
    assert e[n // 2] == pytest.approx(0.5, rel=1e-12)
    assert np.sum(e) == pytest.approx(0.5, rel=1e-12)


def test_vector_field_sums_components():
    grid_shape = (16, 16, 16)
    x, y, z = _coords(grid_shape)
    u = np.stack([np.cos(2 * y), np.sin(2 * z), np.cos(x + y + z)])  # shells 2, 2, 2
    _, e = energy_spectrum(_rfft(u, 3), grid_shape)
    assert e[2] == pytest.approx(3 * 0.25, rel=1e-12)


@pytest.mark.parametrize("k_power", [-2.0, 1.0, 2.0])
def test_k_power_weighting(k_power):
    grid_shape = (32, 32)
    x, y = _coords(grid_shape)
    u = np.cos(3 * x + 4 * y) + 2.0  # mean must not contribute when k_power != 0
    _, s = shell_spectrum(_rfft(u, 2), grid_shape, k_power=k_power)
    assert s[5] == pytest.approx(0.25 * 5.0**k_power, rel=1e-12)
    assert s[0] == 0.0
    assert np.sum(s) == pytest.approx(s[5], rel=1e-12)


def test_single_precision_input_accumulates_in_float64(rng):
    grid_shape = (32, 32)
    u = rng.standard_normal(grid_shape)
    u_hat = _rfft(u, 2).astype(np.complex64)
    _, e = energy_spectrum(u_hat, grid_shape)
    assert e.dtype == np.float64
    assert np.sum(e) == pytest.approx(0.5 * np.mean(u**2), rel=1e-6)


@pytest.mark.parametrize(
    ("shape_hat", "grid_shape"),
    [
        ((32, 17), (32, 30)),  # last axis inconsistent
        ((32, 17), (30, 32)),  # first axis inconsistent
        ((17,), (32, 32)),  # too few axes
    ],
)
def test_shape_mismatch_raises(shape_hat, grid_shape):
    with pytest.raises(ValueError, match="rfft layout"):
        energy_spectrum(np.zeros(shape_hat, dtype=complex), grid_shape)


@pytest.mark.parametrize("grid_shape", [(32,), (4, 4, 4, 4), (0, 8)])
def test_bad_grid_shape_raises(grid_shape):
    with pytest.raises(ValueError, match="grid_shape"):
        shell_mode_count(grid_shape)
