r"""Wavenumber grid and dealiasing mask for the rfft layout (CORE, owner-written).

Conventions (see ``CLAUDE.md``): periodic box :math:`[0, 2\pi)^d`, so wavenumbers are
integers; fields are transformed with ``scipy.fft.rfftn(..., norm="forward")``, giving
arrays of shape ``(N_0, ..., N_{d-2}, N_{d-1} // 2 + 1)`` in Fourier space. Solvers
assume every :math:`N_i` is even.

References
----------
Pope, *Turbulent Flows*, §6.4 (Fourier modes) and §9.1 (pseudo-spectral DNS of
homogeneous turbulence, aliasing and the 2/3 rule).
"""

import numpy as np
from numpy.typing import NDArray


# CORE
def wavenumbers(shape: tuple[int, ...]) -> tuple[NDArray[np.float64], ...]:
    r"""Integer wavenumber components on the rfft layout, for :math:`d = 2, 3`.

    A real field on the grid :math:`x_j = 2\pi j / N` is represented as

    .. math::

        u(\mathbf{x}) = \sum_{\mathbf{k}} \hat u_{\mathbf{k}}\,
        e^{i \mathbf{k}\cdot\mathbf{x}}, \qquad
        k_i \in \{-N_i/2, \dots, N_i/2 - 1\},

    with only :math:`k_{d-1} \ge 0` stored (Hermitian symmetry
    :math:`\hat u_{-\mathbf{k}} = \hat u_{\mathbf{k}}^*`).

    Parameters
    ----------
    shape : tuple of int
        Real-space grid shape ``(N_0, ..., N_{d-1})``, ``d`` in {2, 3}.

    Returns
    -------
    tuple of ndarray
        ``d`` float64 arrays ``(k_0, ..., k_{d-1})``, each broadcastable to the rfft
        shape ``(N_0, ..., N_{d-1} // 2 + 1)``. Along the full axes the ordering is the
        ``fft`` ordering (index ``N_i / 2`` holds the Nyquist wavenumber
        :math:`-N_i/2`); along the last axis :math:`k_{d-1} = 0, \dots, N_{d-1}/2`.

    References
    ----------
    Pope, *Turbulent Flows*, §6.4.
    """
    raise NotImplementedError


# CORE
def k_squared(k: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
    r"""Squared wavenumber magnitude :math:`|\mathbf{k}|^2 = \sum_i k_i^2`.

    Fourier symbol of :math:`-\nabla^2`: :math:`\widehat{\nabla^2 u} = -|\mathbf{k}|^2 \hat u`.

    Parameters
    ----------
    k : tuple of ndarray
        Wavenumber components as returned by :func:`wavenumbers`.

    Returns
    -------
    ndarray
        Dense float64 array with the full rfft shape.

    References
    ----------
    Pope, *Turbulent Flows*, §6.4.
    """
    raise NotImplementedError


# CORE
def dealias_mask(shape: tuple[int, ...]) -> NDArray[np.bool_]:
    r"""Orszag 2/3-rule dealiasing mask on the rfft layout.

    A quadratic product of two fields truncated to :math:`|k_i| \le K` contains modes up
    to :math:`2K`; on an :math:`N`-point grid these alias onto :math:`2K - N`. Aliases
    stay outside the retained band iff :math:`2K - N < -K`, i.e. :math:`K < N/3`. The
    mask therefore keeps

    .. math::

        |k_i| < \frac{N_i}{3} \quad \text{for every } i

    (box truncation), and removes everything else, including the Nyquist modes.

    Parameters
    ----------
    shape : tuple of int
        Real-space grid shape ``(N_0, ..., N_{d-1})``, ``d`` in {2, 3}.

    Returns
    -------
    ndarray of bool
        Dense mask with the rfft shape; ``True`` for retained modes.

    References
    ----------
    Pope, *Turbulent Flows*, §9.1 (aliasing errors and their removal).
    """
    raise NotImplementedError
