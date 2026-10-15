r"""Mixing-length RANS for fully developed turbulent channel flow (CORE, owner-written).

References
----------
Pope, *Turbulent Flows*, §7.1 (channel flow, law of the wall, log law) and §10.3
(mixing-length model, van Driest damping). Guyon, Hulin & Petit, *Hydrodynamique
physique*, ch. 12 (turbulent wall flows, logarithmic law).
"""

import numpy as np
from numpy.typing import NDArray

#: Outer cap on the mixing length, :math:`\ell \le \lambda\,\delta` (Escudier's value).
OUTER_MIXING_LENGTH_RATIO = 0.09


# CORE
def solve_channel(
    re_tau: float,
    n_points: int,
    kappa: float = 0.41,
    a_plus: float = 26.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Mean velocity profile of a channel flow from a mixing-length closure.

    In wall units (:math:`y^+ = y u_\tau/\nu`, :math:`u^+ = \langle U\rangle/u_\tau`,
    half-height :math:`\delta^+ = Re_\tau`), the integrated mean momentum balance gives
    the total shear stress, viscous plus Reynolds:

    .. math::

        \left(1 + \nu_t^+\right)\frac{du^+}{dy^+} = 1 - \frac{y^+}{Re_\tau},
        \qquad
        \nu_t^+ = \ell^{+2}\left|\frac{du^+}{dy^+}\right|,

    with the van Driest-damped mixing length capped in the outer layer,

    .. math::

        \ell^+ = \min\!\left[\kappa y^+\left(1 - e^{-y^+/A^+}\right),\;
        \lambda\, Re_\tau\right], \qquad \lambda = 0.09,

    and :math:`u^+(0) = 0`. For :math:`y^+ \lesssim 5` this yields :math:`u^+ \simeq y^+`;
    for :math:`y^+ \gtrsim 30`, :math:`y/\delta \ll 1`, the log law
    :math:`u^+ \simeq \kappa^{-1}\ln y^+ + B`.

    Parameters
    ----------
    re_tau : float
        Friction Reynolds number :math:`Re_\tau = u_\tau\delta/\nu`.
    n_points : int
        Number of grid points between the wall and the centreline (inclusive).
    kappa : float, optional
        von Kármán constant :math:`\kappa`.
    a_plus : float, optional
        van Driest damping constant :math:`A^+`.

    Returns
    -------
    y_plus : ndarray
        Increasing wall distances, ``y_plus[0] == 0``, ``y_plus[-1] == re_tau``.
    u_plus : ndarray
        Mean velocity :math:`u^+(y^+)`, ``u_plus[0] == 0``.

    References
    ----------
    Pope, *Turbulent Flows*, §7.1, §10.3.
    """
    raise NotImplementedError
