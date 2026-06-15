"""Permutation-invariant code constructors and PIQS embeddings.

This module builds the logical codewords used by the package. Most constructors
first create logical kets in the fully symmetric Dicke top block, whose
dimension is ``N + 1`` for ``N`` qubits. The matching ``*_piqs`` helpers embed
those top-block states into the PIQS reduced space by padding the lower spin
blocks with zeros.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional, Tuple
import qutip


def piqs_block_dims(N: int) -> List[int]:
    """
    Return the PIQS reduced-space block dimensions.

    Parameters
    ----------
    N : int
        Number of physical qubits.

    Returns
    -------
    list[int]
        Block dimensions in descending order, starting with the symmetric
        block ``N + 1``.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    min_dim = 1 if (N % 2 == 0) else 2
    return list(range(N + 1, min_dim - 1, -2))


def piqs_total_dim(N: int) -> int:
    """
    Return the total dimension of the PIQS reduced space.

    Parameters
    ----------
    N : int
        Number of physical qubits.

    Returns
    -------
    int
        Sum of all PIQS block dimensions.
    """
    return sum(piqs_block_dims(N))


def embed_into_piqs_reduced(ket_top: np.ndarray, N: int) -> np.ndarray:
    """
    Embed a top-block ket into the PIQS reduced space.

    Parameters
    ----------
    ket_top : numpy.ndarray
        Ket in the symmetric top block with length ``N + 1``.
    N : int
        Number of physical qubits.

    Returns
    -------
    numpy.ndarray
        Column vector with shape ``(D, 1)``, where ``D`` is the PIQS reduced
        dimension and lower blocks are filled with zeros.
    """
    ket_top = np.asarray(ket_top, dtype=complex).reshape(-1)
    if ket_top.size != N + 1:
        raise ValueError(f"Expected ket_top length N+1={N+1}, got {ket_top.size}")

    D = piqs_total_dim(N)
    ket = np.zeros((D, 1), dtype=complex)
    ket[: N + 1, 0] = ket_top
    return ket


def embed_rho_into_piqs_reduced(rho_top: np.ndarray, N: int) -> np.ndarray:
    """
    Embed a top-block density matrix into the PIQS reduced space.

    Parameters
    ----------
    rho_top : numpy.ndarray
        Density matrix supported on the symmetric top block, with shape
        ``(N + 1, N + 1)``.
    N : int
        Number of physical qubits.

    Returns
    -------
    numpy.ndarray
        Density matrix with shape ``(D, D)``, where ``D`` is the PIQS reduced
        dimension and lower blocks are filled with zeros.
    """
    rho_top = np.asarray(rho_top, dtype=complex)
    if rho_top.shape != (N + 1, N + 1):
        raise ValueError(f"Expected rho_top shape {(N+1, N+1)}, got {rho_top.shape}")

    D = piqs_total_dim(N)
    rho = np.zeros((D, D), dtype=complex)
    rho[: N + 1, : N + 1] = rho_top
    return rho


def code_maximally_mixed_on_codespace(ket0: np.ndarray, ket1: np.ndarray) -> np.ndarray:
    """
    Return the maximally mixed state on a two-dimensional codespace.

    Parameters
    ----------
    ket0, ket1 : numpy.ndarray
        Logical code kets spanning the codespace.

    Returns
    -------
    numpy.ndarray
        Density matrix ``0.5 * (|0_L><0_L| + |1_L><1_L|)``.
    """
    ket0 = np.asarray(ket0, dtype=complex).reshape(-1)
    ket1 = np.asarray(ket1, dtype=complex).reshape(-1)

    n0 = np.linalg.norm(ket0)
    n1 = np.linalg.norm(ket1)
    if n0 == 0 or n1 == 0:
        raise ValueError("One of the kets has zero norm.")

    ket0 = ket0 / n0
    ket1 = ket1 / n1
    return 0.5 * (np.outer(ket0, np.conjugate(ket0)) + np.outer(ket1, np.conjugate(ket1)))


def as_qutip_objects(rho: np.ndarray, ket0: np.ndarray, ket1: np.ndarray):
    """
    Convert code arrays into QuTiP objects.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix array.
    ket0, ket1 : numpy.ndarray
        Logical ket arrays.

    Returns
    -------
    tuple[qutip.Qobj, qutip.Qobj, qutip.Qobj]
        ``(rho, ket0, ket1)`` as QuTiP objects with compatible dims.
    """
    import qutip  # type: ignore

    D = rho.shape[0]
    rhoQ = qutip.Qobj(rho, dims=[[D], [D]])
    l0Q = qutip.Qobj(ket0, dims=[[D], [1]])
    l1Q = qutip.Qobj(ket1, dims=[[D], [1]])
    return rhoQ, l0Q, l1Q


def symmetric_code_to_piqs_reduced(
    ket0_top,
    ket1_top,
    N: int,
    *,
    return_qutip: bool = True,
):
    """
    Embed any symmetric-top-block code into the PIQS reduced space.

    Parameters
    ----------
    ket0_top, ket1_top : array-like or qutip.Qobj
        Logical kets in the symmetric top block with length ``N + 1``.
    N : int
        Number of physical qubits.
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return raw NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` in the PIQS reduced space, as either QuTiP
        objects or NumPy arrays.
    """
    ket0_top = ket0_top.full().ravel() if isinstance(ket0_top, qutip.Qobj) else ket0_top
    ket1_top = ket1_top.full().ravel() if isinstance(ket1_top, qutip.Qobj) else ket1_top
    
    ket0_top = np.asarray(ket0_top, dtype=complex).reshape(-1)
    ket1_top = np.asarray(ket1_top, dtype=complex).reshape(-1)
    if ket0_top.size != N + 1 or ket1_top.size != N + 1:
        raise ValueError(f"Expected both kets length {N+1}")

    # Normalize (safe, even if already normalized)
    ket0_top = ket0_top / np.linalg.norm(ket0_top)
    ket1_top = ket1_top / np.linalg.norm(ket1_top)

    rho_top = code_maximally_mixed_on_codespace(ket0_top, ket1_top)
    rho = embed_rho_into_piqs_reduced(rho_top, N)

    l0 = embed_into_piqs_reduced(ket0_top, N)
    l1 = embed_into_piqs_reduced(ket1_top, N)

    if return_qutip:
        return as_qutip_objects(rho, l0, l1)
    return rho, l0, l1



def symmetric_basis_state(N: int, w: int) -> np.ndarray:
    """
    Return a Dicke basis vector in the symmetric top block.

    Parameters
    ----------
    N : int
        Number of physical qubits.
    w : int
        Dicke weight index in ``0, ..., N``.

    Returns
    -------
    numpy.ndarray
        Length ``N + 1`` vector representing ``|D_w^N>``.
    """
    if not (0 <= w <= N):
        raise ValueError(f"Require 0<=w<=N, got w={w}, N={N}")
    v = np.zeros(N + 1, dtype=complex)
    v[w] = 1.0
    return v


def double_factorial(n: int) -> int:
    """
    Compute a non-negative integer double factorial.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    int
        Double factorial ``n!! = n (n - 2) (n - 4) ...``.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n in (0, 1):
        return 1
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out

# ============================================================
# (g,n,u) - PI code
# ============================================================
def gnucode_kets_in_top_block(
    g: int,
    n: int,
    u: int,
    *,
    return_qutip: bool = False,
    qutip_dims: Optional[list] = None,
):
    """
    Construct top-block logical kets for a ``(g, n, u)`` GNU PI code.

    Parameters
    ----------
    g : int
        Dicke-weight spacing parameter.
    n : int
        Binomial order parameter.
    u : int
        Qubit-number scale parameter, with ``N = g * n * u``.
    return_qutip : bool, default=False
        If true, return QuTiP kets. If false, return NumPy arrays.
    qutip_dims : list, optional
        Custom QuTiP dims used when ``return_qutip`` is true.

    Returns
    -------
    tuple
        ``(ket0, ket1, N)`` where the kets are normalized top-block logical
        states.
    """
    if g <= 0 or n < 0 or u <= 0:
        raise ValueError("Require g>0, n>=0, u>0")
    N = int(g * n * u)

    psi0 = np.zeros(int(N + 1), dtype=complex)
    psi1 = np.zeros(int(N + 1), dtype=complex)

    # Build unnormalized logical kets in symmetric subspace
    for ell in range(n + 1):
        w = g * ell
        if w > N:
            continue
        amp = np.sqrt(math.comb(n, ell))
        if (ell % 2) == 0:
            psi0[w] += amp
        else:
            psi1[w] += amp

    # Normalize each logical state
    n0 = np.linalg.norm(psi0)
    n1 = np.linalg.norm(psi1)
    if n0 == 0 or n1 == 0:
        raise ValueError("One of the logical states became zero. Check parameters g,n,u.")
    psi0 /= n0
    psi1 /= n1

    if not return_qutip:
        return psi0, psi1, N

    # QuTiP output (ket Qobj in (N+1)-dim symmetric subspace)
    try:
        from qutip import Qobj
    except ImportError as e:
        raise ImportError("return_qutip=True but qutip is not installed/importable.") from e

    dims = qutip_dims if qutip_dims is not None else [[N + 1], [1]]
    ket0 = Qobj(psi0.reshape(-1, 1), dims=dims)
    ket1 = Qobj(psi1.reshape(-1, 1), dims=dims)
    return ket0, ket1, N

def gnucode_piqs(g: int, n: int, u: int, *, return_qutip: bool = True):
    """
    Return the PIQS reduced-space embedding of a ``(g, n, u)`` GNU code.

    Parameters
    ----------
    g : int
        Dicke-weight spacing parameter.
    n : int
        Binomial order parameter.
    u : int
        Qubit-number scale parameter.
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` embedded in the PIQS reduced space.
    """
    ket0_top, ket1_top, N = gnucode_kets_in_top_block(g, n, u, return_qutip=False)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)



# ============================================================
# (b,g)-PI code
# ============================================================

def bgcode_kets_in_top_block(b: int, g: int, return_qutip: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construct top-block logical kets for a ``(b, g)`` PI code.

    Parameters
    ----------
    b : int
        Code parameter controlling Dicke weights.
    g : int
        Code parameter controlling Dicke weights.
    return_qutip : bool, default=False
        If true, return QuTiP kets. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(ket0_top, ket1_top, N)`` where ``N = 2 * b + g``.
    """
    if b <= 0 or g <= 0:
        raise ValueError("b and g must be positive integers.")
    if 2 * b < g + 1:
        raise ValueError("Constraint 2b >= g+1 must hold.")

    N = 2 * b + g

    D0  = symmetric_basis_state(N, 0)
    D2b = symmetric_basis_state(N, 2 * b)
    Dg  = symmetric_basis_state(N, g)
    DN  = symmetric_basis_state(N, N)

    ket0 = (math.sqrt(2*b - g) * D0 + math.sqrt(2*b + g) * D2b) / math.sqrt(4*b)
    ket1 = (math.sqrt(2*b - g) * DN + math.sqrt(2*b + g) * Dg) / math.sqrt(4*b)

    ket0 = ket0 / np.linalg.norm(ket0)
    ket1 = ket1 / np.linalg.norm(ket1)
    if return_qutip:
        import qutip  # type: ignore
        ket0 = qutip.Qobj(ket0, dims=[[N + 1], [1]])
        ket1 = qutip.Qobj(ket1, dims=[[N + 1], [1]])
    return ket0, ket1, N


def bgcode_piqs(b: int, g: int, *, return_qutip: bool = True):
    """
    Return the PIQS reduced-space embedding of a ``(b, g)`` PI code.

    Parameters
    ----------
    b : int
        Code parameter controlling Dicke weights.
    g : int
        Code parameter controlling Dicke weights.
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` embedded in the PIQS reduced space.
    """
    ket0_top, ket1_top, N = bgcode_kets_in_top_block(b, g)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)


# ============================================================
# (b,g,m)-PI code
# ============================================================

def bgm_gamma_vectors(b: int, g: int, m: int) -> np.ndarray:
    """
    Return the gamma coefficients for a ``(b, g, m)`` PI code.

    Parameters
    ----------
    b : int
        Code parameter controlling Dicke weights.
    g : int
        Code parameter controlling Dicke weights.
    m : int
        Code order parameter.

    Returns
    -------
    numpy.ndarray
        Array ``gamma[k]`` for ``k = 0, ..., m``.
    """
    if b <= 0 or g <= 0 or m <= 0:
        raise ValueError("b, g, m must be positive integers.")

    gamma = np.zeros(m + 1, dtype=float)
    prefactor = b ** (-m / 2)

    for k in range(m + 1):
        prod1 = 1.0
        for i in range(k + 1, m + 1):
            prod1 *= math.sqrt(2 * i * b - g)

        prod2 = 1.0
        for j in range(m - k + 1, m + 1):
            prod2 *= math.sqrt(2 * j * b + g)

        gamma[k] = prefactor * prod1 * prod2

    return gamma


def bgmcode_kets_in_top_block(b: int, g: int, m: int, return_qutip: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construct top-block logical kets for a ``(b, g, m)`` PI code.

    Parameters
    ----------
    b : int
        Code parameter controlling Dicke weights.
    g : int
        Code parameter controlling Dicke weights.
    m : int
        Code order parameter, with ``N = 2 * b * m + g``.
    return_qutip : bool, default=False
        If true, return QuTiP kets. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(ket0_top, ket1_top, N)`` for the normalized top-block logical
        states.
    """
    if b <= 0 or g <= 0 or m <= 0:
        raise ValueError("b, g, m must be positive integers.")

    N = 2 * b * m + g
    ket0 = np.zeros(N + 1, dtype=complex)

    gamma = bgm_gamma_vectors(b, g, m)
    norm_den = (2 ** m) * math.sqrt(double_factorial(2 * m - 1))

    for k in range(m + 1):
        w = 2 * b * k
        amp = math.sqrt(math.comb(m, k)) * gamma[k] / norm_den
        ket0[w] = amp

    ket0 = ket0 / np.linalg.norm(ket0)
    ket1 = ket0[::-1].copy()  # X^{⊗N}: w -> N-w
    if return_qutip:
        import qutip  # type: ignore
        ket0 = qutip.Qobj(ket0, dims=[[N + 1], [1]])
        ket1 = qutip.Qobj(ket1, dims=[[N + 1], [1]])
    return ket0, ket1, N


def bgmcode_piqs(b: int, g: int, m: int, *, return_qutip: bool = True):
    """
    Return the PIQS reduced-space embedding of a ``(b, g, m)`` PI code.

    Parameters
    ----------
    b : int
        Code parameter controlling Dicke weights.
    g : int
        Code parameter controlling Dicke weights.
    m : int
        Code order parameter.
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` embedded in the PIQS reduced space.
    """
    ket0_top, ket1_top, N = bgmcode_kets_in_top_block(b, g, m)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)


# ===============================================================
# 7-qubit Pollatsek-Ruskai PI code
# ===============================================================
def pollatsek_ruskai_7_kets_in_top_block(
    epsilon: int = 1,
    return_qutip: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construct top-block kets for the 7-qubit Pollatsek-Ruskai PI code.

    Parameters
    ----------
    epsilon : int, default=1
        Sign parameter, either ``+1`` or ``-1``.
    return_qutip : bool, default=False
        If true, return QuTiP kets. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(ket0_top, ket1_top, 7)`` for the normalized top-block logical
        states.
    """
    if epsilon not in (-1, 1):
        raise ValueError("epsilon must be +1 or -1.")

    N = 7
    eps = float(epsilon)
    ket0_top = (
        eps * math.sqrt(15) * symmetric_basis_state(N, 0)
        - math.sqrt(7) * symmetric_basis_state(N, 2)
        + eps * math.sqrt(21) * symmetric_basis_state(N, 4)
        + math.sqrt(21) * symmetric_basis_state(N, 6)
    ) / 8
    ket1_top = (
        math.sqrt(21) * symmetric_basis_state(N, 1)
        + eps * math.sqrt(21) * symmetric_basis_state(N, 3)
        - math.sqrt(7) * symmetric_basis_state(N, 5)
        + eps * math.sqrt(15) * symmetric_basis_state(N, 7)
    ) / 8

    ket0_top = ket0_top / np.linalg.norm(ket0_top)
    ket1_top = ket1_top / np.linalg.norm(ket1_top)

    if return_qutip:
        import qutip  # type: ignore
        ket0_top = qutip.Qobj(ket0_top, dims=[[N + 1], [1]])
        ket1_top = qutip.Qobj(ket1_top, dims=[[N + 1], [1]])
    return ket0_top, ket1_top, N


def pollatsek_ruskai_7_piqs(epsilon: int = 1, *, return_qutip: bool = True):
    """
    Return the PIQS embedding of the 7-qubit Pollatsek-Ruskai PI code.

    Parameters
    ----------
    epsilon : int, default=1
        Sign parameter, either ``+1`` or ``-1``.
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` embedded in the PIQS reduced space.
    """
    ket0_top, ket1_top, N = pollatsek_ruskai_7_kets_in_top_block(epsilon=epsilon)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)


# ===============================================================
# 11-qubit Kubischta-Teixeira PI code
# ===============================================================
def kubischta_teixeira_11_kets_in_top_block(
    return_qutip: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construct top-block kets for the 11-qubit Kubischta-Teixeira PI code.

    Parameters
    ----------
    return_qutip : bool, default=False
        If true, return QuTiP kets. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(ket0_top, ket1_top, 11)`` for the normalized top-block logical
        states.
    """
    N = 11
    ket0_top = (
        math.sqrt(5) * symmetric_basis_state(N, 0)
        + math.sqrt(11) * symmetric_basis_state(N, 8)
    ) / 4
    ket1_top = (
        math.sqrt(11) * symmetric_basis_state(N, 3)
        + math.sqrt(5) * symmetric_basis_state(N, 11)
    ) / 4

    ket0_top = ket0_top / np.linalg.norm(ket0_top)
    ket1_top = ket1_top / np.linalg.norm(ket1_top)

    if return_qutip:
        import qutip  # type: ignore
        ket0_top = qutip.Qobj(ket0_top, dims=[[N + 1], [1]])
        ket1_top = qutip.Qobj(ket1_top, dims=[[N + 1], [1]])
    return ket0_top, ket1_top, N


def kubischta_teixeira_11_piqs(*, return_qutip: bool = True):
    """
    Return the PIQS embedding of the 11-qubit Kubischta-Teixeira PI code.

    Parameters
    ----------
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` embedded in the PIQS reduced space.
    """
    ket0_top, ket1_top, N = kubischta_teixeira_11_kets_in_top_block()
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)

# ===============================================================
# 7 qubit PI code 
# ===============================================================
def seven_qubit_code_kets_in_top_block(return_qutip: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construct top-block kets for the 7-qubit PI code.

    Parameters
    ----------
    return_qutip : bool, default=False
        If true, return QuTiP kets. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(ket0_top, ket1_top, 7)`` for the normalized top-block logical
        states.
    """
    N = 7
    ket0_top = np.sqrt(3/10)*symmetric_basis_state(N,0) + np.sqrt(7/10)*symmetric_basis_state(N,5)
    ket1_top = np.sqrt(7/10)*symmetric_basis_state(N,2) - np.sqrt(3/10)*symmetric_basis_state(N,7)
    if return_qutip:
        import qutip  # type: ignore
        ket0_top = qutip.Qobj(ket0_top, dims=[[N + 1], [1]])
        ket1_top = qutip.Qobj(ket1_top, dims=[[N + 1], [1]])
    return ket0_top, ket1_top, N

def seven_qubit_piqs(*, return_qutip: bool = True):
    """
    Return the PIQS embedding of the 7-qubit PI code.

    Parameters
    ----------
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` embedded in the PIQS reduced space.
    """
    ket0_top, ket1_top, N = seven_qubit_code_kets_in_top_block(return_qutip=return_qutip)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)

# ============================================================
# Gross 13-qubit code (top block dim 14, reduced dim 56)
# ============================================================

def gross_13_kets_in_top_block(phi: float = 0.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construct top-block kets for the Gross 13-qubit code.

    Parameters
    ----------
    phi : float, default=0.0
        Relative phase used in the Gross 13-qubit construction.

    Returns
    -------
    tuple[qutip.Qobj, qutip.Qobj, int]
        ``(ket0_top, ket1_top, 13)`` as normalized QuTiP kets in the top block.
    """
    N = 13
    j = N / 2
    dim = N + 1

    c1 = math.sqrt(910) / 56
    c2 = -3 * math.sqrt(154) / 56
    c3 = -math.sqrt(770) / 56
    c4 = math.sqrt(70) / 56

    c5 = math.sqrt(231) / 84
    c6 = math.sqrt(1365) / 84
    c7 = -math.sqrt(273) / 28
    c8 = -math.sqrt(3003) / 84

    # QuTiP spin basis ordering: m = j, j-1, ..., -j  (descending)
    # index = j - m
    def ket_jm(m: float) -> qutip.Qobj:
        """
        Return a QuTiP spin-basis ket for a magnetic quantum number.

        Parameters
        ----------
        m : float
            Magnetic quantum number in the ``j = N / 2`` spin basis.

        Returns
        -------
        qutip.Qobj
            Basis ket in QuTiP's descending-``m`` ordering.
        """
        idx = int(round(j - m))  # 0-based index in QuTiP ordering
        return qutip.basis(dim, idx)

    # These m values match your MATLAB construction
    b1 = ket_jm(+13/2)   # m=+6.5 -> idx 0
    b2 = ket_jm(+5/2)    # m=+2.5 -> idx 4
    b3 = ket_jm(-3/2)    # m=-1.5 -> idx 8
    b4 = ket_jm(-11/2)   # m=-5.5 -> idx 12

    state1 = c1*b1 + c2*b2 + c3*b3 + c4*b4
    state2 = c5*b1 + c6*b2 + c7*b3 + c8*b4

    phase = np.exp(1j * phi)
    a = np.sqrt(13/7) / 2
    b = np.sqrt(1 - a**2)

    ket0 = a * state1 + b * phase * state2
    ket0 = ket0.unit()  # normalize (QuTiP)

    # logical-1 by applying exp(-i pi Jx)
    Jx = qutip.jmat(j, 'x')          # Qobj operator dim x dim
    U = (-1j * np.pi * Jx).expm()
    ket1 = (U * ket0).unit()

    # Convert to numpy 1D arrays at the end
    #ket0_np = ket0.full().ravel()
    #ket1_np = ket1.full().ravel()

    return ket0, ket1, N



def gross_13_piqs(phi: float = 0.0, *, return_qutip: bool = True):
    """
    Return the PIQS embedding of the Gross 13-qubit code.

    Parameters
    ----------
    phi : float, default=0.0
        Relative phase used in the Gross 13-qubit construction.
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` embedded in the PIQS reduced space.
    """
    ket0_top, ket1_top, N = gross_13_kets_in_top_block(phi)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)


# ============================================================
# Gross 17-qubit code (top block dim 18, reduced dim 90)
# ============================================================

def gross_17_kets_in_top_block() -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construct top-block kets for the Gross 17-qubit code.

    Parameters
    ----------
    None
        This constructor has no user-supplied parameters.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, int]
        ``(ket0_top, ket1_top, 17)`` for the normalized top-block logical
        states.
    """
    N = 17
    s = math.sqrt

    ket0_top = np.array([
        0,
        (1/25056)*((-2)*s(663*(4619-34*s(17290))) + 9*s(3570*(281+2*s(17290)))),
        0, 0, 0,
        (1/12528)*(4*s(357*(4619-34*s(17290))) - 3*s(6630*(281+2*s(17290)))),
        0, 0, 0,
        (1/4176)*s(187/87)*(s(290*(4619-34*s(17290))) + s(2639*(281+2*s(17290)))),
        0, 0, 0,
        (1/72)*s((17/6)*(281+2*s(17290))),
        0, 0, 0,
        (1/25056)*(22*s(39*(4619-34*s(17290))) + 17*s(210*(281+2*s(17290)))),
    ], dtype=complex)

    ket1_top = np.array([
        (17/13824)*s((35/6)*(281+2*s(17290)))
        + (43/3207168)*(22*s(39*(4619-34*s(17290))) + 17*s(210*(281+2*s(17290))))
        + (187/801792)*s(65/174)*(s(290*(4619-34*s(17290))) + s(2639*(281+2*s(17290))))
        + (1/9621504)*s(17)*((-2)*s(663*(4619-34*s(17290))) + 9*s(3570*(281+2*s(17290))))
        + (1/2405376)*s(1547)*(4*s(357*(4619-34*s(17290))) - 3*s(6630*(281+2*s(17290)))),
        0, 0, 0,
        (29/6912)*s((17/6)*(281+2*s(17290)))
        + (1/4810752)*s(595)*(22*s(39*(4619-34*s(17290))) + 17*s(210*(281+2*s(17290))))
        + (11/400896)*s(1547/174)*(s(290*(4619-34*s(17290))) + s(2639*(281+2*s(17290))))
        + (1/534528)*s(35)*((-2)*s(663*(4619-34*s(17290))) + 9*s(3570*(281+2*s(17290))))
        - (1/400896)*s(65)*(4*s(357*(4619-34*s(17290))) - 3*s(6630*(281+2*s(17290)))),
        0, 0, 0,
        (1/13824)*s((17017/3)*(281+2*s(17290)))
        + (1/4810752)*s(12155/2)*(22*s(39*(4619-34*s(17290))) + 17*s(210*(281+2*s(17290))))
        + (11/89088)*s(187/87)*(s(290*(4619-34*s(17290))) + s(2639*(281+2*s(17290))))
        + (1/4810752)*s(715/2)*((-2)*s(663*(4619-34*s(17290))) + 9*s(3570*(281+2*s(17290))))
        + (1/1202688)*s(385/2)*(4*s(357*(4619-34*s(17290))) - 3*s(6630*(281+2*s(17290)))),
        0, 0, 0,
        (-1/2304)*s((1105/6)*(281+2*s(17290)))
        + (1/4810752)*s(1547)*(22*s(39*(4619-34*s(17290))) + 17*s(210*(281+2*s(17290))))
        + (11/400896)*s(595/174)*(s(290*(4619-34*s(17290))) + s(2639*(281+2*s(17290))))
        - (7/4810752)*s(91)*((-2)*s(663*(4619-34*s(17290))) + 9*s(3570*(281+2*s(17290))))
        + (53/1202688)*(4*s(357*(4619-34*s(17290))) - 3*s(6630*(281+2*s(17290)))),
        0, 0, 0,
        (1/1536)*s((595/6)*(281+2*s(17290)))
        + (1/9621504)*s(17)*(22*s(39*(4619-34*s(17290))) + 17*s(210*(281+2*s(17290))))
        + (11/801792)*s(1105/174)*(s(290*(4619-34*s(17290))) + s(2639*(281+2*s(17290))))
        + (113/9621504)*((-2)*s(663*(4619-34*s(17290))) + 9*s(3570*(281+2*s(17290))))
        - (7/2405376)*s(91)*(4*s(357*(4619-34*s(17290))) - 3*s(6630*(281+2*s(17290)))),
        0
    ], dtype=complex)

    # normalize
    ket0_top = ket0_top / np.linalg.norm(ket0_top)
    ket1_top = ket1_top / np.linalg.norm(ket1_top)

    if ket0_top.size != N + 1 or ket1_top.size != N + 1:
        raise RuntimeError("gross_17 vectors are not length 18; check transcription.")

    return ket0_top, ket1_top, N


def gross_17_piqs(*, return_qutip: bool = True):
    """
    Return the PIQS embedding of the Gross 17-qubit code.

    Parameters
    ----------
    return_qutip : bool, default=True
        If true, return QuTiP objects. If false, return NumPy arrays.

    Returns
    -------
    tuple
        ``(rho, ket0, ket1)`` embedded in the PIQS reduced space.
    """
    ket0_top, ket1_top, N = gross_17_kets_in_top_block()
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)
