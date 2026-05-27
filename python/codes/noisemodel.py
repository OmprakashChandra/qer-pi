import numpy as np
import qutip
from qutip.piqs.piqs import*
from qutip import*
from typing import List, Union
import qutip.piqs as piqs


def noisemodel(
    noise_type: str,
    num_qubits: int,
    gamma: float,
    dt: float,
    return_rep: str = "kraus",  # "kraus" | "choi" | "super"
    dynamics: str = "exact",    # "exact" | "approx"
):
    """
    Generate a quantum noise channel in PIQS and return either Kraus operators,
    the Choi matrix, or the superoperator.

    Parameters
    ----------
    noise_type : str
        Supported: 'global symmetric depolarizing', 'local symmetric depolarizing', 'global symmetric amplitude damping', 'local symmetric amplitude damping'
    num_qubits : int
        Number of qubits.
    gamma : float
        Noise rate parameter.
    dt : float
        Time step.
    return_rep : str
        'kraus' -> list of Kraus operators (slow: Choi eigen-decomposition)
        'choi'  -> Choi matrix (fast)
        'super' -> superoperator
    dynamics : str
        'exact'  -> use the exact PIQS Lindbladian dynamics exp(L t)
        'approx' -> use the second-order Kraus approximation from Eq. 16.
                    Currently only supported for global symmetric amplitude damping.

    Returns
    -------
    If return_rep == 'kraus': list[qutip.Qobj]
    If return_rep == 'choi' : qutip.Qobj
    If return_rep == 'super': qutip.Qobj
    """
    noise_type = noise_type.lower().strip()
    return_rep = return_rep.lower().strip()
    dynamics = dynamics.lower().strip()

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

    raise ValueError(
        "Unsupported noise type. Supported types: "
        "'global symmetric depolarizing', 'local symmetric depolarizing', "
        "'global symmetric amplitude damping', 'local symmetric amplitude damping'."
    )


def _require_exact_dynamics(noise_type: str, dynamics: str):
    if dynamics != "exact":
        raise ValueError(
            f"dynamics='{dynamics}' is only implemented for "
            "'global symmetric amplitude damping'. "
            f"Use dynamics='exact' for '{noise_type}'."
        )


def _return_from_kraus(kraus: List[qutip.Qobj], return_rep: str):
    if return_rep == "kraus":
        return kraus
    if return_rep == "choi":
        return qutip.kraus_to_choi(kraus)
    if return_rep == "super":
        return qutip.kraus_to_super(kraus)
    raise ValueError("return_rep must be 'kraus', 'choi', or 'super'.")


def _global_symmetric_amplitude_damping_approx_kraus(
    num_qubits: int,
    gamma: float,
    dt: float,
) -> List[qutip.Qobj]:
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
    ident = qutip.qeye(Jm.dims[0])
    Nop = Jp * Jm

    K0 = ident - (alpha / 2) * Nop + (alpha**2 / 8) * (Nop * Nop)
    K1 = np.sqrt(alpha) * Jm - (alpha**1.5 / 4) * (Jm * Nop + Nop * Jm)
    K2 = (alpha / np.sqrt(2)) * (Jm * Jm)

    K3 = ident - K0.dag() * K0 - K1.dag() * K1 - K2.dag() * K2

    return [K0, K1, K2, K3]


def _global_symmetric_depolarizing(num_qubits: int, gamma: float, dt: float, return_rep: str):
    """Global symmetric depolarizing-like noise using PIQS collective rates."""
    system = Dicke(num_qubits)
    system.collective_emission  = gamma
    system.collective_pumping   = gamma
    system.collective_dephasing = gamma

    L = system.liouvillian()
    E = (L * dt).expm()  # superoperator for the CPTP map

    if return_rep == "super":
        return E

    choi = to_choi(E)

    if return_rep == "choi":
        return choi

    if return_rep == "kraus":
        d = int(np.sqrt(L.shape[0]))  # since L is d^2 x d^2
        return _kraus_from_choi(choi, d_out=d, d_in=d)

    raise ValueError("return_rep must be 'kraus', 'choi', or 'super'.")


def _local_symmetric_depolarizing(num_qubits: int, gamma: float, dt: float, return_rep: str):
    """Local symmetric depolarizing-like noise using PIQS local rates."""
    system = Dicke(num_qubits)
    system.emission  = gamma
    system.pumping   = gamma
    system.dephasing = gamma

    L = system.liouvillian()
    E = (L * dt).expm()

    if return_rep == "super":
        return E

    choi = to_choi(E)

    if return_rep == "choi":
        return choi

    if return_rep == "kraus":
        d = int(np.sqrt(L.shape[0]))
        return _kraus_from_choi(choi, d_out=d, d_in=d)

    raise ValueError("return_rep must be 'kraus', 'choi', or 'super'.")

def _global_symmetric_amplitude_damping(
    num_qubits: int,
    gamma: float,
    dt: float,
    return_rep: str,
    dynamics: str = "exact",
): 
    """ generates a superoperator corresponding to Lindbladian with jump operator J_ = J_x - iJ_y, where Jx and Jy are the collective spin operators of N+1 dimensions. """
    if dynamics == "approx":
        kraus = _global_symmetric_amplitude_damping_approx_kraus(num_qubits, gamma, dt)
        return _return_from_kraus(kraus, return_rep)

    if dynamics != "exact":
        raise ValueError("dynamics must be either 'exact' or 'approx'.")

    system = Dicke(num_qubits)
    system.collective_emission = gamma
    L = system.liouvillian()
    E = (L * dt).expm()    
    if return_rep == "super":
        return E 
    if return_rep == "choi":
        choi = to_choi(E) 
        return choi
    if return_rep == "kraus": 
        d = int(np.sqrt(L.shape[0]))
        choi = to_choi(E)
        return _kraus_from_choi(choi, d_out = d, d_in = d)
    raise ValueError("return_rep must be 'kraus', 'choi', or 'super'.")

def _local_symmetric_amplitude_damping(num_qubits: int, gamma: float, dt: float, return_rep: str):
    """ generates a superoperator corresponding to Lindbladian with jump operators J_-^(i) = sigma_-^(i) for i=1,...,N, where sigma_-^(i) is the lowering operator on the i-th qubit. """
    system = Dicke(num_qubits)
    system.emission = gamma
    L = system.liouvillian()
    E = (L * dt).expm()    
    if return_rep == "super":
        return E 
    choi = to_choi(E)
    if return_rep == "choi": 
        return choi
    if return_rep == "kraus": 
        d = int(np.sqrt(L.shape[0]))
        return _kraus_from_choi(choi, d_out = d, d_in = d)
    raise ValueError("return_rep must be 'kraus', 'choi', or 'super'.")


def _kraus_from_choi(choi: qutip.Qobj, d_out: int, d_in: int, tol: float = 1e-15) -> List[qutip.Qobj]:
    """
    Convert a Choi matrix to a list of Kraus operators using eigen-decomposition.
    This is the slow step for large dimensions.

    Kraus operators returned have shape (d_out x d_in).
    """
    choi = (choi + choi.dag()) / 2  # enforce Hermiticity
    w, v = np.linalg.eigh(choi.full())

    Ks: List[qutip.Qobj] = []
    for k in range(len(w)):
        if w[k] <= tol:
            continue
        vec = np.sqrt(w[k]) * v[:, k]
        Kmat = vec.reshape((d_out, d_in), order="F")  # column-stacking
        Ks.append(qutip.Qobj(Kmat, dims=[[d_out], [d_in]]))

    return Ks
