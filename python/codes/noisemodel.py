import numpy as np
import qutip
from qutip.piqs.piqs import*
from qutip import*
from typing import List, Union


def noisemodel(
    noise_type: str,
    num_qubits: int,
    gamma: float,
    dt: float,
    return_rep: str = "kraus",  # "kraus" | "choi" | "super"
):
    """
    Generate a quantum noise channel in PIQS and return either Kraus operators,
    the Choi matrix, or the superoperator S = exp(L t).

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
        'super' -> superoperator S = exp(L t)

    Returns
    -------
    If return_rep == 'kraus': list[qutip.Qobj]
    If return_rep == 'choi' : qutip.Qobj
    If return_rep == 'super': qutip.Qobj
    """
    noise_type = noise_type.lower().strip()
    return_rep = return_rep.lower().strip()

    if noise_type == "global symmetric depolarizing":
        return _global_symmetric_depolarizing(num_qubits, gamma, dt, return_rep)

    if noise_type == "local symmetric depolarizing":
        return _local_symmetric_depolarizing(num_qubits, gamma, dt, return_rep)

    if noise_type == "global symmetric amplitude damping": 
        return _global_symmetric_amplitude_damping(num_qubits, gamma, dt, return_rep)

    if noise_type == "local symmetric amplitude damping":
        return _local_symmetric_amplitude_damping(num_qubits, gamma, dt, return_rep)

    raise ValueError(
        "Unsupported noise type. Supported types: "
        "'global symmetric depolarizing', 'local symmetric depolarizing', "
        "'global symmetric amplitude damping', 'local symmetric amplitude damping'."
    )


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

def _global_symmetric_amplitude_damping(num_qubits: int, gamma: float, dt: float, return_rep: str): 
    """ generates a superoperator corresponding to Lindbladian with jump operator J_ = J_x - iJ_y, where Jx and Jy are the collective spin operators of N+1 dimensions. """
    system = Dicke(num_qubits)
    system.collective_emission = gamma
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

    
