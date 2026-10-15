r"""Shell-summed spectra from ``rfftn`` coefficients (2D and 3D).

Conventions (``CLAUDE.md``): periodic box :math:`[0, 2\pi)^d`, integer wavenumbers,
``scipy.fft.rfftn(u, norm="forward")``, so the :math:`\hat u_{\mathbf{k}}` are true Fourier
coefficients and Parseval reads

.. math::

    \langle u^2 \rangle = \sum_{\mathbf{k}\in\mathbb{Z}^d} |\hat u_{\mathbf{k}}|^2
    = \sum_{\text{rfft half}} w_{\mathbf{k}} |\hat u_{\mathbf{k}}|^2,

where the half-spectrum weight :math:`w_{\mathbf{k}}` counts how many full-spectrum
modes a stored coefficient stands for: 2 for :math:`0 < k_{d-1} < N_{d-1}/2` (the mode
and its Hermitian conjugate :math:`-\mathbf{k}`, which is not stored), 1 on the
:math:`k_{d-1} = 0` and Nyquist :math:`k_{d-1} = N_{d-1}/2` planes (both members of each
conjugate pair are stored there).

Shell :math:`n` collects the modes with :math:`n - \tfrac12 \le |\mathbf{k}| < n + \tfrac12`.
Since :math:`|\mathbf{k}|^2` is an integer, :math:`|\mathbf{k}|` is never a half-integer,
so no mode sits on a shell boundary.

This module is deliberately independent of :mod:`cascade.grid` (CORE): it builds its own
integer wavenumbers, used only for binning.

References
----------
Pope, *Turbulent Flows*, §6.5 (velocity spectra, :math:`E(k)`).
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "energy_spectrum",
    "half_spectrum_weights",
    "shell_mode_count",
    "shell_spectrum",
]


def _rfft_shape(grid_shape: tuple[int, ...]) -> tuple[int, ...]:
    return (*grid_shape[:-1], grid_shape[-1] // 2 + 1)


def _validate_grid_shape(grid_shape: tuple[int, ...]) -> tuple[int, ...]:
    grid_shape = tuple(int(n) for n in grid_shape)
    if len(grid_shape) not in (2, 3):
        raise ValueError(f"grid_shape must have 2 or 3 entries, got {grid_shape}")
    if any(n < 1 for n in grid_shape):
        raise ValueError(f"grid_shape entries must be positive, got {grid_shape}")
    return grid_shape


def _readonly(a: NDArray) -> NDArray:
    a.setflags(write=False)
    return a


@lru_cache(maxsize=8)
def _k2_int(grid_shape: tuple[int, ...]) -> NDArray[np.int64]:
    """Integer :math:`|k|^2` on the rfft layout (exact, no floating-point round-off)."""
    d = len(grid_shape)
    k2 = np.zeros(_rfft_shape(grid_shape), dtype=np.int64)
    for axis, n in enumerate(grid_shape):
        if axis < d - 1:
            # fft ordering: 0, 1, ..., ceil(n/2) - 1, -floor(n/2), ..., -1
            k = np.concatenate((np.arange(0, (n + 1) // 2), np.arange(-(n // 2), 0)))
        else:
            k = np.arange(0, n // 2 + 1)
        bshape = [1] * d
        bshape[axis] = k.size
        k2 += (k * k).reshape(bshape)
    return _readonly(k2)


@lru_cache(maxsize=8)
def _shell_index(grid_shape: tuple[int, ...]) -> NDArray[np.intp]:
    """Shell index ``rint(|k|)`` of every stored mode, flattened (C order)."""
    kmag = np.sqrt(_k2_int(grid_shape).astype(np.float64))
    return _readonly(np.rint(kmag).astype(np.intp).ravel())


@lru_cache(maxsize=8)
def _half_weights_1d(grid_shape: tuple[int, ...]) -> NDArray[np.float64]:
    n = grid_shape[-1]
    w = np.full(n // 2 + 1, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0  # Nyquist plane: stored once, its conjugate is in the same plane
    return _readonly(w)


def half_spectrum_weights(grid_shape: tuple[int, ...]) -> NDArray[np.float64]:
    r"""Multiplicity :math:`w_{\mathbf{k}}` of each stored rfft coefficient.

    Parameters
    ----------
    grid_shape : tuple of int
        Real-space grid shape ``(N_0, ..., N_{d-1})``, ``d`` in {2, 3}. The last entry is
        needed explicitly: ``N // 2 + 1`` does not tell an even ``N`` (Nyquist plane,
        weight 1) from an odd one (no Nyquist plane, weight 2).

    Returns
    -------
    ndarray
        Weights of shape ``(1, ..., 1, N_{d-1} // 2 + 1)``, broadcastable to the rfft
        layout: 1 at :math:`k_{d-1} = 0` and (``N_{d-1}`` even) at
        :math:`k_{d-1} = N_{d-1}/2`, 2 otherwise. Read-only.

    Examples
    --------
    >>> half_spectrum_weights((8, 8)).ravel()
    array([1., 2., 2., 2., 1.])
    >>> half_spectrum_weights((7, 7)).ravel()
    array([1., 2., 2., 2.])
    """
    grid_shape = _validate_grid_shape(grid_shape)
    w = _half_weights_1d(grid_shape)
    return w.reshape((1,) * (len(grid_shape) - 1) + (w.size,))


def shell_mode_count(grid_shape: tuple[int, ...]) -> NDArray[np.float64]:
    r"""Number of full-spectrum modes :math:`N_n` in each shell.

    Useful to compensate the lattice-counting jitter of the shell sums at low :math:`k`,
    e.g. :math:`E(k)\,2\pi k / N_n` (2D) or :math:`E(k)\,4\pi k^2 / N_n` (3D) for a
    shell-averaged estimate, at the cost of exact Parseval.

    Parameters
    ----------
    grid_shape : tuple of int
        Real-space grid shape, ``d`` in {2, 3}.

    Returns
    -------
    ndarray
        :math:`N_n` for :math:`n = 0, \dots, n_{\max}`; sums to ``prod(grid_shape)``.
    """
    grid_shape = _validate_grid_shape(grid_shape)
    w = np.broadcast_to(half_spectrum_weights(grid_shape), _rfft_shape(grid_shape))
    return np.bincount(_shell_index(grid_shape), weights=w.ravel())


def shell_spectrum(
    f_hat: NDArray[np.complexfloating],
    grid_shape: tuple[int, ...],
    k_power: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Shell-summed spectrum of a scalar or vector field.

    .. math::

        S(n) = \frac12 \sum_{n - \frac12 \le |\mathbf{k}| < n + \frac12}
        |\mathbf{k}|^{p} \sum_c |\hat f_{c,\mathbf{k}}|^2
        \qquad (\text{full spectrum } \mathbf{k}\in\mathbb{Z}^d),

    evaluated on the stored half spectrum with weights :math:`w_{\mathbf{k}}` (see
    :func:`half_spectrum_weights`). With ``k_power = 0``,
    :math:`\sum_n S(n) = \tfrac12 \langle |\mathbf{f}|^2 \rangle` exactly.

    Parameters
    ----------
    f_hat : ndarray of complex
        Coefficients from ``scipy.fft.rfftn(f, axes=..., norm="forward")``; the last
        ``d`` axes are the rfft layout of ``grid_shape``, leading axes (if any) are field
        components and are summed over.
    grid_shape : tuple of int
        Real-space grid shape ``(N_0, ..., N_{d-1})``, ``d`` in {2, 3}.
    k_power : float, optional
        Exponent :math:`p` of an extra :math:`|\mathbf{k}|^p` factor, e.g. ``-2`` for the
        energy spectrum from 2D vorticity, ``2`` for the enstrophy spectrum from velocity.
        For :math:`p \ne 0` the :math:`\mathbf{k} = 0` mode contributes nothing.

    Returns
    -------
    k : ndarray
        Shell wavenumbers :math:`0, 1, \dots, n_{\max}` (float64), with
        :math:`n_{\max} = \mathrm{rint}\,|\mathbf{k}|_{\max}` (the box corner), so that no
        mode is discarded.
    spectrum : ndarray
        :math:`S(n)` (float64), same length as ``k``.

    Raises
    ------
    ValueError
        If the trailing axes of ``f_hat`` do not match the rfft layout of ``grid_shape``.
    """
    grid_shape = _validate_grid_shape(grid_shape)
    d = len(grid_shape)
    rshape = _rfft_shape(grid_shape)
    f_hat = np.asarray(f_hat)
    if f_hat.ndim < d or f_hat.shape[-d:] != rshape:
        raise ValueError(
            f"f_hat trailing shape {f_hat.shape[-d:]} does not match the rfft layout "
            f"{rshape} of grid_shape {grid_shape}"
        )

    # Σ_c |f̂_c|², accumulated in float64 whatever the input precision.
    f = f_hat.reshape((-1, *rshape))
    re = f.real.astype(np.float64, copy=False)
    im = f.imag.astype(np.float64, copy=False)
    power = np.sum(re * re + im * im, axis=0)
    power *= 0.5 * half_spectrum_weights(grid_shape)
    if k_power != 0:
        k2 = _k2_int(grid_shape)
        factor = np.zeros(rshape)
        nonzero = k2 > 0
        factor[nonzero] = k2[nonzero].astype(np.float64) ** (0.5 * k_power)
        power *= factor

    spectrum = np.bincount(_shell_index(grid_shape), weights=power.ravel())
    k = np.arange(spectrum.size, dtype=np.float64)
    return k, spectrum


def energy_spectrum(
    u_hat: NDArray[np.complexfloating],
    grid_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Kinetic-energy spectrum :math:`E(k)` from velocity coefficients.

    .. math::

        E(n) = \frac12 \sum_{n - \frac12 \le |\mathbf{k}| < n + \frac12}
        |\hat{\mathbf u}_{\mathbf{k}}|^2, \qquad
        \sum_n E(n) = \tfrac12\langle|\mathbf{u}|^2\rangle,

    the discrete analogue of :math:`E(k) = \oint_{|\mathbf{k}|=k}
    \tfrac12 \hat u_i^* \hat u_i\, dS` with unit shell width (Pope §6.5).

    Parameters
    ----------
    u_hat : ndarray of complex
        Velocity coefficients, shape ``(d, *rfft_shape)`` (or any leading component
        axes), from ``rfftn(u, axes=range(-d, 0), norm="forward")``.
    grid_shape : tuple of int
        Real-space grid shape, ``d`` in {2, 3}.

    Returns
    -------
    k : ndarray
        Shell wavenumbers :math:`0, \dots, n_{\max}`.
    E : ndarray
        :math:`E(n)`.

    See Also
    --------
    shell_spectrum : general form (scalar fields, :math:`|\mathbf{k}|^p` weighting).
    """
    return shell_spectrum(u_hat, grid_shape)
