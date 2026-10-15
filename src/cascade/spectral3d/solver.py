r"""Pseudo-spectral 3D incompressible Navier–Stokes, rotational form (CORE, owner-written).

Governing equations on :math:`[0, 2\pi)^3`, with :math:`\boldsymbol\omega = \nabla\times\mathbf{u}`:

.. math::

    \partial_t \mathbf{u} = \mathbf{u}\times\boldsymbol\omega - \nabla\Pi
    + \nu\nabla^2\mathbf{u} + \mathbf{f}, \qquad \nabla\cdot\mathbf{u} = 0, \qquad
    \Pi = p/\rho + \tfrac12|\mathbf{u}|^2 .

In Fourier space (``rfftn``, ``norm="forward"``) the pressure is eliminated by the Leray
projector :math:`P_{ij}(\mathbf{k}) = \delta_{ij} - k_i k_j / |\mathbf{k}|^2`:

.. math::

    \partial_t \hat u_i = -\nu |\mathbf{k}|^2 \hat u_i
    + P_{ij}(\mathbf{k})\,\widehat{(\mathbf{u}\times\boldsymbol\omega)}_j + \hat f_i .

Arrays of vector fields carry the component on the leading axis: shape
``(3, N, N, N // 2 + 1)``, ``N`` even.

References
----------
Pope, *Turbulent Flows*, §6.4 (Navier–Stokes in wavenumber space, projection tensor),
§9.1 (pseudo-spectral DNS of homogeneous turbulence). Guyon, Hulin & Petit,
*Hydrodynamique physique*, ch. 12.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


# CORE
def leray_project(
    v_hat: ComplexArray,
    k: tuple[RealArray, RealArray, RealArray],
    k2: RealArray,
) -> ComplexArray:
    r"""Project a vector field onto its divergence-free part.

    .. math::

        (\mathcal{P}\hat{\mathbf v})_i = \hat v_i - \frac{k_i\, k_j \hat v_j}{|\mathbf{k}|^2},
        \qquad \mathbf{k}\cdot\mathcal{P}\hat{\mathbf v} = 0 ,

    the :math:`\mathbf{k} = 0` (mean) mode being left unchanged.

    Parameters
    ----------
    v_hat : ndarray of complex
        Vector field, shape ``(3, N, N, N // 2 + 1)``.
    k : tuple of ndarray
        ``(kx, ky, kz)`` from :func:`cascade.grid.wavenumbers`.
    k2 : ndarray
        :math:`|\mathbf{k}|^2` from :func:`cascade.grid.k_squared`.

    Returns
    -------
    ndarray of complex
        :math:`\mathcal{P}\hat{\mathbf v}` (new array).

    References
    ----------
    Pope, *Turbulent Flows*, §6.4.
    """
    raise NotImplementedError


# CORE
def nonlinear_rhs(
    u_hat: ComplexArray,
    k: tuple[RealArray, RealArray, RealArray],
    k2: RealArray,
    mask: NDArray[np.bool_],
) -> ComplexArray:
    r"""Dealiased, projected nonlinear term in rotational form.

    .. math::

        \hat N_i = \mathcal{M}_{\mathbf{k}}\, P_{ij}(\mathbf{k})\,
        \mathcal{F}\!\left[\mathbf{u}\times\boldsymbol\omega\right]_j, \qquad
        \hat{\boldsymbol\omega} = i\mathbf{k}\times\hat{\mathbf{u}},

    with the product formed in physical space and :math:`\mathcal{M}` the 2/3 mask. The
    result is solenoidal, :math:`\mathbf{k}\cdot\hat{\mathbf N} = 0`, and, for a dealiased
    solenoidal input, conserves kinetic energy exactly:
    :math:`\sum_{\mathbf{k}} \mathrm{Re}(\hat{\mathbf u}^*\cdot\hat{\mathbf N}) = 0`
    to round-off (since :math:`\mathbf{u}\cdot(\mathbf{u}\times\boldsymbol\omega) = 0`).

    Parameters
    ----------
    u_hat : ndarray of complex
        Velocity, shape ``(3, N, N, N // 2 + 1)``.
    k : tuple of ndarray
        ``(kx, ky, kz)``.
    k2 : ndarray
        :math:`|\mathbf{k}|^2`.
    mask : ndarray of bool
        2/3 mask from :func:`cascade.grid.dealias_mask`.

    Returns
    -------
    ndarray of complex
        :math:`\hat{\mathbf N}`, same shape as ``u_hat``, zero outside ``mask``.

    References
    ----------
    Pope, *Turbulent Flows*, §6.4, §9.1.
    """
    raise NotImplementedError


# CORE
def rk4_if_step(
    u_hat: ComplexArray,
    dt: float,
    lin: RealArray,
    nonlinear: Callable[[ComplexArray], ComplexArray],
) -> ComplexArray:
    r"""Integrating-factor RK4 (Lawson) step for :math:`\dot{\hat u} = L\hat u + N`.

    For Navier–Stokes :math:`L_{\mathbf{k}} = -\nu|\mathbf{k}|^2`. With
    :math:`\hat v = e^{-Lt}\hat u`,

    .. math::

        \frac{d\hat v}{dt} = e^{-L t}\, N\!\left[e^{L t}\hat v\right],

    integrated by classical RK4: viscous decay is exact, global error
    :math:`O(\Delta t^4)`. Same contract as :func:`cascade.spectral2d.solver.rk4_if_step`
    (the owner may share one implementation).

    Parameters
    ----------
    u_hat : ndarray of complex
        State at :math:`t_n`.
    dt : float
        Time step.
    lin : ndarray
        Diagonal linear operator, broadcastable to ``u_hat`` (e.g. shape
        ``(N, N, N // 2 + 1)`` against ``(3, N, N, N // 2 + 1)``).
    nonlinear : callable
        :math:`\hat u \mapsto N[\hat u]`, e.g. :func:`nonlinear_rhs` plus
        :func:`constant_power_forcing`.

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
def constant_power_forcing(
    u_hat: ComplexArray,
    k2: RealArray,
    k_f: float,
    epsilon: float,
) -> ComplexArray:
    r"""Deterministic forcing injecting a constant power :math:`\varepsilon` at low :math:`k`.

    .. math::

        \hat{\mathbf f}_{\mathbf{k}} = \frac{\varepsilon}{2E_f}\,\hat{\mathbf u}_{\mathbf{k}}
        \quad (0 < |\mathbf{k}| \le k_f), \qquad
        E_f = \tfrac12 \sum_{0<|\mathbf{k}|\le k_f} |\hat{\mathbf u}_{\mathbf{k}}|^2,

    and zero elsewhere, so that (full-spectrum sums)

    .. math::

        \sum_{\mathbf{k}} \mathrm{Re}\left(\hat{\mathbf u}^*_{\mathbf{k}}\cdot
        \hat{\mathbf f}_{\mathbf{k}}\right) = \varepsilon

    exactly at every instant. The forcing is parallel to :math:`\hat{\mathbf u}`, hence
    solenoidal. In a statistically steady state :math:`\varepsilon` equals the mean
    dissipation, which fixes the Kolmogorov scale :math:`\eta = (\nu^3/\varepsilon)^{1/4}`
    a priori.

    Parameters
    ----------
    u_hat : ndarray of complex
        Velocity, shape ``(3, N, N, N // 2 + 1)``.
    k2 : ndarray
        :math:`|\mathbf{k}|^2`.
    k_f : float
        Forcing cut-off wavenumber.
    epsilon : float
        Injected power :math:`\varepsilon`.

    Returns
    -------
    ndarray of complex
        :math:`\hat{\mathbf f}`, same shape as ``u_hat``.

    References
    ----------
    Pope, *Turbulent Flows*, §9.1 (forcing). Lundgren, CTR Annual Research Briefs (2003)
    (linear forcing, of which this is the band-limited, constant-power variant).
    """
    raise NotImplementedError


# CORE
def random_solenoidal_field(
    shape: tuple[int, int, int],
    k_peak: float,
    energy: float,
    rng: np.random.Generator,
) -> ComplexArray:
    r"""Random, real, solenoidal initial velocity with a prescribed energy spectrum.

    Random phases with modal amplitudes chosen so that the shell spectrum follows

    .. math::

        E(k) \propto k^4 \exp\!\left(-2\,k^2/k_p^2\right),

    then projected (:math:`\mathbf{k}\cdot\hat{\mathbf u} = 0`) and normalised to

    .. math::

        \tfrac12 \sum_{\mathbf{k}} |\hat{\mathbf u}_{\mathbf{k}}|^2
        = \tfrac12 \langle|\mathbf{u}|^2\rangle = E_0 .

    The field is Hermitian-consistent (``rfftn(irfftn(u_hat)) == u_hat``), has zero mean
    and vanishes outside the 2/3 dealiasing band.

    Parameters
    ----------
    shape : tuple of int
        Real-space grid shape ``(N, N, N)``, ``N`` even.
    k_peak : float
        Peak wavenumber :math:`k_p` of :math:`E(k)`.
    energy : float
        Total kinetic energy :math:`E_0`.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    ndarray of complex
        :math:`\hat{\mathbf u}`, shape ``(3, N, N, N // 2 + 1)``.

    References
    ----------
    Pope, *Turbulent Flows*, §6.5 (model spectrum), §9.1 (initial conditions).
    """
    raise NotImplementedError
