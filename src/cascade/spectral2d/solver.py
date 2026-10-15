r"""Pseudo-spectral 2D Navier–Stokes in vorticity–streamfunction form (CORE, owner-written).

Governing equation on :math:`[0, 2\pi)^2`, with :math:`\omega = \partial_x v - \partial_y u
= -\nabla^2 \psi`, :math:`u = \partial_y \psi`, :math:`v = -\partial_x \psi`:

.. math::

    \partial_t \omega + \mathbf{u}\cdot\nabla\omega
    = -\nu_p (-\nabla^2)^{p}\, \omega - \mu\, \omega + f_\omega .

In Fourier space (``rfft2``, ``norm="forward"``) this splits into a stiff, diagonal linear
part and a nonlinear part,

.. math::

    \partial_t \hat\omega_{\mathbf{k}} = L_{\mathbf{k}} \hat\omega_{\mathbf{k}}
    + \hat N_{\mathbf{k}}[\hat\omega] + \hat f_{\mathbf{k}}, \qquad
    L_{\mathbf{k}} = -\nu_p |\mathbf{k}|^{2p} - \mu, \qquad
    \hat N = -\widehat{(\mathbf{u}\cdot\nabla\omega)} .

Forced 2D turbulence (Kraichnan double cascade): energy flows to :math:`k < k_f` and is
removed by the drag :math:`\mu`; enstrophy flows to :math:`k > k_f` and is removed by the
hyperviscosity :math:`\nu_p`.

References
----------
Guyon, Hulin & Petit, *Hydrodynamique physique*, ch. 12 (2D turbulence, double cascade).
Pope, *Turbulent Flows*, §9.1 (pseudo-spectral method). Kraichnan, Phys. Fluids 10, 1417
(1967). Boffetta & Ecke, Annu. Rev. Fluid Mech. 44, 427 (2012).
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


# CORE
def nonlinear_rhs(
    omega_hat: ComplexArray,
    kx: RealArray,
    ky: RealArray,
    k2: RealArray,
    mask: NDArray[np.bool_],
) -> ComplexArray:
    r"""Dealiased nonlinear term of the vorticity equation.

    .. math::

        \hat N_{\mathbf{k}} = -\,\mathcal{M}_{\mathbf{k}}\,
        \mathcal{F}\!\left[u\,\partial_x\omega + v\,\partial_y\omega\right]_{\mathbf{k}},
        \qquad
        \hat\psi = \frac{\hat\omega}{|\mathbf{k}|^2}\ (\hat\psi_{\mathbf 0} = 0),\quad
        \hat u = i k_y \hat\psi,\quad \hat v = -i k_x \hat\psi,

    where :math:`\mathcal{M}` is the 2/3 mask. Products are evaluated in physical space
    (pseudo-spectral). With a dealiased input the truncated system conserves energy
    :math:`E = \tfrac12 \sum |\hat\omega|^2/k^2` and enstrophy
    :math:`Z = \tfrac12 \sum |\hat\omega|^2` exactly: :math:`\sum \hat\psi^* \hat N` and
    :math:`\sum \hat\omega^* \hat N` vanish to round-off.

    Parameters
    ----------
    omega_hat : ndarray of complex
        Vorticity coefficients, shape ``(N, N // 2 + 1)``, ``N`` even.
    kx, ky : ndarray
        Wavenumber components from :func:`cascade.grid.wavenumbers`.
    k2 : ndarray
        :math:`|\mathbf{k}|^2` from :func:`cascade.grid.k_squared`.
    mask : ndarray of bool
        2/3 mask from :func:`cascade.grid.dealias_mask`.

    Returns
    -------
    ndarray of complex
        :math:`\hat N`, same shape as ``omega_hat``, zero outside ``mask``.

    References
    ----------
    Pope, *Turbulent Flows*, §9.1. Guyon, Hulin & Petit, ch. 12.
    """
    raise NotImplementedError


# CORE
def linear_operator(k2: RealArray, nu: float, mu: float, order: int = 1) -> RealArray:
    r"""Diagonal linear operator: (hyper)viscosity plus linear drag.

    .. math::

        L_{\mathbf{k}} = -\nu_p |\mathbf{k}|^{2p} - \mu .

    ``order = 1`` is ordinary viscosity :math:`\nu\nabla^2`; ``order = p > 1`` is
    hyperviscosity :math:`-\nu_p(-\nabla^2)^p`, which confines dissipation to the
    smallest resolved scales. The drag :math:`\mu` (Ekman friction) removes the energy
    of the inverse cascade at large scales.

    Parameters
    ----------
    k2 : ndarray
        :math:`|\mathbf{k}|^2`.
    nu : float
        (Hyper)viscosity coefficient :math:`\nu_p \ge 0`.
    mu : float
        Linear drag coefficient :math:`\mu \ge 0`.
    order : int, optional
        Exponent :math:`p \ge 1` of :math:`(-\nabla^2)^p`.

    Returns
    -------
    ndarray
        :math:`L_{\mathbf{k}}` (real, :math:`\le 0`), same shape as ``k2``.

    References
    ----------
    Boffetta & Ecke, Annu. Rev. Fluid Mech. 44, 427 (2012), §2.
    """
    raise NotImplementedError


# CORE
def rk4_if_step(
    omega_hat: ComplexArray,
    dt: float,
    lin: RealArray,
    nonlinear: Callable[[ComplexArray], ComplexArray],
) -> ComplexArray:
    r"""Integrating-factor RK4 (Lawson) step for :math:`\dot{\hat\omega} = L\hat\omega + N`.

    With :math:`\hat v(t) = e^{-L t}\,\hat\omega(t)` the stiff linear term is removed
    exactly,

    .. math::

        \frac{d\hat v}{dt} = e^{-L t}\, N\!\left[e^{L t}\hat v\right],

    and the classical fourth-order Runge–Kutta scheme is applied to :math:`\hat v`
    over :math:`[t_n, t_n + \Delta t]`. Linear decay is therefore exact for any
    :math:`\Delta t`; the global error is :math:`O(\Delta t^4)`.

    Parameters
    ----------
    omega_hat : ndarray of complex
        State at :math:`t_n`.
    dt : float
        Time step :math:`\Delta t`.
    lin : ndarray
        Diagonal linear operator :math:`L`, broadcastable to ``omega_hat``
        (see :func:`linear_operator`).
    nonlinear : callable
        :math:`\hat\omega \mapsto N[\hat\omega]`, e.g. a closure over
        :func:`nonlinear_rhs`. Must not include white-in-time forcing (see
        :func:`ring_forcing`).

    Returns
    -------
    ndarray of complex
        State at :math:`t_n + \Delta t` (new array).

    References
    ----------
    Lawson, SIAM J. Numer. Anal. 4, 372 (1967). Pope, *Turbulent Flows*, §9.1.
    """
    raise NotImplementedError


# CORE
def cfl_time_step(
    omega_hat: ComplexArray,
    kx: RealArray,
    ky: RealArray,
    k2: RealArray,
    cfl: float = 0.5,
    dt_max: float = np.inf,
) -> float:
    r"""Advective CFL time step.

    .. math::

        \Delta t = \min\!\left(\Delta t_{\max},\
        \frac{C}{\max_{\mathbf{x}}\left(|u|/\Delta x + |v|/\Delta y\right)}\right),
        \qquad \Delta x = \frac{2\pi}{N_x},\ \Delta y = \frac{2\pi}{N_y},

    with the velocity evaluated on the collocation grid. Viscous and drag terms impose
    no limit since the integrating factor treats them exactly.

    Parameters
    ----------
    omega_hat : ndarray of complex
        Vorticity coefficients, shape ``(N_x, N_y // 2 + 1)``, ``N_y`` even.
    kx, ky, k2 : ndarray
        Wavenumbers and :math:`|\mathbf{k}|^2`.
    cfl : float, optional
        Courant number :math:`C`.
    dt_max : float, optional
        Upper bound, returned when the flow is at rest.

    Returns
    -------
    float
        :math:`\Delta t`.

    References
    ----------
    Pope, *Turbulent Flows*, §9.1.
    """
    raise NotImplementedError


# CORE
def ring_forcing(
    k2: RealArray,
    k_f: float,
    delta_k: float,
    epsilon: float,
    dt: float,
    rng: np.random.Generator,
) -> ComplexArray:
    r"""White-in-time Gaussian forcing increment on a wavenumber ring.

    The ring is :math:`k_f - \delta k \le |\mathbf{k}| \le k_f + \delta k`.

    The forcing :math:`\hat f_{\mathbf{k}}(t)` is Gaussian, zero-mean, delta-correlated in
    time and supported on the ring. Over one step it adds the increment

    .. math::

        \Delta\hat\omega_{\mathbf{k}} = \int_{t_n}^{t_n+\Delta t} \hat f_{\mathbf{k}}\,dt
        \;\propto\; \sqrt{\Delta t}\;\xi_{\mathbf{k}},

    normalised so that the mean energy injection rate is exactly :math:`\varepsilon`,
    independent of the flow (white noise is uncorrelated with :math:`\hat\omega`):

    .. math::

        \left\langle \tfrac12 \sum_{\mathbf{k}}
        \frac{|\Delta\hat\omega_{\mathbf{k}}|^2}{|\mathbf{k}|^2} \right\rangle
        = \varepsilon\, \Delta t,

    (sum over the full spectrum). Enstrophy is then injected at
    :math:`\eta \approx k_f^2 \varepsilon`. The increment is added after the
    deterministic step (first-order splitting); it must respect Hermitian symmetry on
    the :math:`k_y = 0` line so the forced field stays real.

    Parameters
    ----------
    k2 : ndarray
        :math:`|\mathbf{k}|^2` on the rfft layout.
    k_f : float
        Forcing wavenumber.
    delta_k : float
        Ring half-width.
    epsilon : float
        Mean energy injection rate :math:`\varepsilon`.
    dt : float
        Time step :math:`\Delta t`.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    ndarray of complex
        :math:`\Delta\hat\omega`, shape of ``k2``.

    References
    ----------
    Boffetta & Ecke, Annu. Rev. Fluid Mech. 44, 427 (2012), §2. Novikov, Sov. Phys. JETP
    20, 1290 (1965) (injection rate of Gaussian white forcing).
    """
    raise NotImplementedError
