"""Noise-channel constructors in the PIQS reduced Dicke basis.

The public :func:`noisemodel` helper builds common permutation-invariant noise
channels for ``num_qubits`` physical qubits and returns the channel as Kraus
operators, a Choi matrix, or a superoperator. Exact channels are generated from
QuTiP/PIQS Lindbladians. The approximate Kraus dynamics are currently only
implemented for global symmetric amplitude damping.
"""

from __future__ import annotations

from typing import List

import numpy as np
import qutip as qt
import qutip.piqs as piqs


__all__ = ["noisemodel"]

SUPPORTED_NOISE_TYPES = (
    "global symmetric depolarizing",
    "local symmetric depolarizing",
    "global symmetric amplitude damping",
    "local symmetric amplitude damping",
)
SUPPORTED_RETURN_REPS = ("kraus", "choi", "super")
FIRST_ORDER_DYNAMICS_ALIASES = {
    "first order ad",
    "first-order ad",
    "first_order_ad",
    "first order",
    "first",
}


def noisemodel(
    noise_type: str,
    num_qubits: int,
    gamma: float,
    dt: float,
    return_rep: str = "kraus",
    dynamics: str = "exact",
):
    """
    Build a PIQS reduced-space noise channel.

    Supported channels are:

    - ``"global symmetric depolarizing"``: collective emission, pumping, and
      dephasing rates all set to ``gamma``.
    - ``"local symmetric depolarizing"``: local emission, pumping, and
      dephasing rates all set to ``gamma``.
    - ``"global symmetric amplitude damping"``: collective emission rate
      ``gamma``.
    - ``"local symmetric amplitude damping"``: local emission rate ``gamma``.

    The exact dynamics use the PIQS Lindbladian map ``exp(L * dt)``. Approximate
    Kraus dynamics are available only for global symmetric amplitude damping.

    Parameters
    ----------
    noise_type : str
        One of the supported channel names listed above. Matching is
        case-insensitive and ignores leading/trailing spaces.
    num_qubits : int
        Number of physical qubits in the PIQS model.
    gamma : float
        Non-negative noise-rate parameter.
    dt : float
        Non-negative evolution time.
    return_rep : str
        Output representation:

        - ``"kraus"``: list of Kraus operators. For exact channels this is the
          slowest option because it diagonalizes the Choi matrix.
        - ``"choi"``: Choi matrix.
        - ``"super"``: superoperator.

    dynamics : str
        Dynamics model:

        - ``"exact"``: PIQS Lindbladian dynamics ``exp(L * dt)``.
        - ``"approx"``: second-order Kraus approximation for global symmetric
          amplitude damping.
        - ``"first order AD"``: first-order Kraus approximation for global
          symmetric amplitude damping.

    Returns
    -------
    list[qutip.Qobj] or qutip.Qobj
        The requested channel representation.
    """
    noise_type = _normalise_noise_type(noise_type)
    return_rep = _normalise_return_rep(return_rep)
    dynamics = _normalise_dynamics(dynamics)
    num_qubits = _validate_channel_parameters(num_qubits, gamma, dt)

    if noise_type == "global symmetric depolarizing":
        _require_exact_dynamics(noise_type, dynamics)
        return _global_symmetric_depolarizing(num_qubits, gamma, dt, return_rep)

    if noise_type == "local symmetric depolarizing":
        _require_exact_dynamics(noise_type, dynamics)
        return _local_symmetric_depolarizing(num_qubits, gamma, dt, return_rep)

    if noise_type == "global symmetric amplitude damping":
        return _global_symmetric_amplitude_damping(num_qubits, gamma, dt, return_rep, dynamics)

    if noise_type == "local symmetric amplitude damping":
        _require_exact_dynamics(noise_type, dynamics)
        return _local_symmetric_amplitude_damping(num_qubits, gamma, dt, return_rep)

    raise AssertionError(f"Unhandled noise type: {noise_type}")


def _normalise_noise_type(noise_type: str) -> str:
    value = noise_type.lower().strip()
    if value not in SUPPORTED_NOISE_TYPES:
        supported = "', '".join(SUPPORTED_NOISE_TYPES)
        raise ValueError(f"Unsupported noise_type='{noise_type}'. Use one of: '{supported}'.")
    return value


def _normalise_return_rep(return_rep: str) -> str:
    value = return_rep.lower().strip()
    if value not in SUPPORTED_RETURN_REPS:
        supported = "', '".join(SUPPORTED_RETURN_REPS)
        raise ValueError(f"return_rep must be one of: '{supported}'.")
    return value


def _normalise_dynamics(dynamics: str) -> str:
    value = dynamics.lower().strip()
    if value in FIRST_ORDER_DYNAMICS_ALIASES:
        return "first order ad"
    if value not in ("exact", "approx"):
        raise ValueError("dynamics must be 'exact', 'approx', or 'first order AD'.")
    return value


def _validate_channel_parameters(num_qubits: int, gamma: float, dt: float) -> int:
    if int(num_qubits) != num_qubits or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if dt < 0:
        raise ValueError("dt must be non-negative.")
    return int(num_qubits)


def _require_exact_dynamics(noise_type: str, dynamics: str):
    if dynamics != "exact":
        raise ValueError(
            f"dynamics='{dynamics}' is only implemented for "
            "'global symmetric amplitude damping'. "
            f"Use dynamics='exact' for '{noise_type}'."
        )


def _return_from_kraus(kraus: List[qt.Qobj], return_rep: str):
    if return_rep == "kraus":
        return kraus
    if return_rep == "choi":
        return qt.kraus_to_choi(kraus)
    if return_rep == "super":
        return qt.kraus_to_super(kraus)
    raise AssertionError(f"Unhandled return_rep: {return_rep}")


def _return_from_superoperator(superoperator: qt.Qobj, return_rep: str):
    if return_rep == "super":
        return superoperator

    choi = qt.to_choi(superoperator)
    if return_rep == "choi":
        return choi

    if return_rep == "kraus":
        d = int(np.sqrt(superoperator.shape[0]))
        return _kraus_from_choi(choi, d_out=d, d_in=d)

    raise AssertionError(f"Unhandled return_rep: {return_rep}")


def _global_symmetric_amplitude_damping_approx_kraus(
    num_qubits: int,
    gamma: float,
    dt: float,
) -> List[qt.Qobj]:
    """
    Eq. 16 second-order Kraus approximation for global symmetric amplitude damping.

    The small parameter is alpha = gamma * dt and J_- is QuTiP/PIQS's
    collective lowering operator in the Dicke reduced basis. K3 is implemented
    literally as the Eq. 16 residual operator, so the map is trace-preserving up
    to the retained second-order dynamics rather than exactly CPTP at finite dt.
    """
    alpha = gamma * dt
    if alpha < 0:
        raise ValueError("gamma * dt must be non-negative for amplitude damping.")

    Jm = piqs.jspin(num_qubits, "-", basis="dicke")
    Jp = Jm.dag()
    ident = qt.qeye(Jm.dims[0])
    Nop = Jp * Jm

    K0 = ident - (alpha / 2) * Nop + (alpha**2 / 8) * (Nop * Nop)
    K1 = np.sqrt(alpha) * Jm - (alpha**1.5 / 4) * (Jm * Nop + Nop * Jm)
    K2 = (alpha / np.sqrt(2)) * (Jm * Jm)

    K3 = ident - K0.dag() * K0 - K1.dag() * K1 - K2.dag() * K2

    return [K0, K1, K2, K3]


def _global_symmetric_amplitude_damping_first_order_kraus(
    num_qubits: int,
    gamma: float,
    dt: float,
) -> List[qt.Qobj]:
    """
    First-order Kraus approximation for global symmetric amplitude damping.

    With alpha = gamma * dt and N = J_+ J_-, the Kraus operators

        K0 = I - alpha N / 2,
        K1 = sqrt(alpha) J_-,

    reproduce I + alpha D[J_-] through linear order in alpha. The resulting
    map is completely positive by construction and trace-preserving up to the
    retained first-order dynamics.
    """
    alpha = gamma * dt
    if alpha < 0:
        raise ValueError("gamma * dt must be non-negative for amplitude damping.")

    Jm = piqs.jspin(num_qubits, "-", basis="dicke")
    Jp = Jm.dag()
    ident = qt.qeye(Jm.dims[0])
    Nop = Jp * Jm

    K0 = ident - (alpha / 2) * Nop
    K1 = np.sqrt(alpha) * Jm

    return [K0, K1]


def _global_symmetric_depolarizing(num_qubits: int, gamma: float, dt: float, return_rep: str):
    """Collective PIQS emission, pumping, and dephasing at equal rates."""
    system = piqs.Dicke(num_qubits)
    system.collective_emission = gamma
    system.collective_pumping = gamma
    system.collective_dephasing = gamma

    L = system.liouvillian()
    return _return_from_superoperator((L * dt).expm(), return_rep)


def _local_symmetric_depolarizing(num_qubits: int, gamma: float, dt: float, return_rep: str):
    """Local PIQS emission, pumping, and dephasing at equal rates."""
    system = piqs.Dicke(num_qubits)
    system.emission = gamma
    system.pumping = gamma
    system.dephasing = gamma

    L = system.liouvillian()
    return _return_from_superoperator((L * dt).expm(), return_rep)


def _global_symmetric_amplitude_damping(
    num_qubits: int,
    gamma: float,
    dt: float,
    return_rep: str,
    dynamics: str = "exact",
):
    """Global amplitude damping with collective lowering jump operator J_-."""
    if dynamics == "first order ad":
        kraus = _global_symmetric_amplitude_damping_first_order_kraus(num_qubits, gamma, dt)
        return _return_from_kraus(kraus, return_rep)

    if dynamics == "approx":
        kraus = _global_symmetric_amplitude_damping_approx_kraus(num_qubits, gamma, dt)
        return _return_from_kraus(kraus, return_rep)

    system = piqs.Dicke(num_qubits)
    system.collective_emission = gamma
    L = system.liouvillian()
    return _return_from_superoperator((L * dt).expm(), return_rep)


def _local_symmetric_amplitude_damping(num_qubits: int, gamma: float, dt: float, return_rep: str):
    """Local amplitude damping with single-qubit lowering jumps sigma_-^(i)."""
    system = piqs.Dicke(num_qubits)
    system.emission = gamma
    L = system.liouvillian()
    return _return_from_superoperator((L * dt).expm(), return_rep)


def _kraus_from_choi(choi: qt.Qobj, d_out: int, d_in: int, tol: float = 1e-15) -> List[qt.Qobj]:
    """
    Convert a Choi matrix to a list of Kraus operators using eigen-decomposition.
    This is the slow step for large dimensions.

    Kraus operators returned have shape (d_out x d_in).
    """
    choi = (choi + choi.dag()) / 2  # enforce Hermiticity
    w, v = np.linalg.eigh(choi.full())

    Ks: List[qt.Qobj] = []
    for k in range(len(w)):
        if w[k] <= tol:
            continue
        vec = np.sqrt(w[k]) * v[:, k]
        Kmat = vec.reshape((d_out, d_in), order="F")  # column-stacking
        Ks.append(qt.Qobj(Kmat, dims=[[d_out], [d_in]]))

    return Ks
