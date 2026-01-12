"""
Permutation-invariant (PI) code constructors + PIQS reduced-space (block-diagonal) embeddings.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple, List


def piqs_block_dims(N: int) -> List[int]:
    """
    Return PIQS reduced-space block dimensions for N qubits:
      [N+1, N-1, N-3, ..., (1 if N even else 2)]
    in descending order (top block first).
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    min_dim = 1 if (N % 2 == 0) else 2
    return list(range(N + 1, min_dim - 1, -2))


def piqs_total_dim(N: int) -> int:
    """Total dimension of PIQS reduced space for N qubits."""
    return sum(piqs_block_dims(N))


def embed_into_piqs_reduced(ket_top: np.ndarray, N: int) -> np.ndarray:
    """
    Embed a ket from the top block (dim N+1) into PIQS reduced space (dim sum_J (2J+1)).
    Output is column vector shape (D, 1), with the top block filled and rest zeros.
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
    Embed a density matrix supported on the top block (shape (N+1,N+1)) into PIQS reduced space (D,D).
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
    rho = 1/2 (|0_L><0_L| + |1_L><1_L|)
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
    Convenience wrapper to return qutip.Qobj objects.
    qutip is imported lazily so this file can be used without qutip installed.
    """
    import qutip  # type: ignore

    D = rho.shape[0]
    rhoQ = qutip.Qobj(rho, dims=[[D], [D]])
    l0Q = qutip.Qobj(ket0, dims=[[D], [1]])
    l1Q = qutip.Qobj(ket1, dims=[[D], [1]])
    return rhoQ, l0Q, l1Q


def symmetric_code_to_piqs_reduced(
    ket0_top: np.ndarray,
    ket1_top: np.ndarray,
    N: int,
    *,
    return_qutip: bool = True,
):
    """
    Generic embedding for any code defined in the top (fully symmetric) block.
    Returns (rho, l0, l1) either as qutip.Qobj or raw numpy arrays.
    """
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
    Dicke basis vector |D_w> in the symmetric subspace of N qubits.
    Vector length N+1, 0-based index w.
    """
    if not (0 <= w <= N):
        raise ValueError(f"Require 0<=w<=N, got w={w}, N={N}")
    v = np.zeros(N + 1, dtype=complex)
    v[w] = 1.0
    return v


def double_factorial(n: int) -> int:
    """Compute n!! = n(n-2)(n-4)..."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if n in (0, 1):
        return 1
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


# ============================================================
# (b,g)-PI code
# ============================================================

def bgcode_kets_in_top_block(b: int, g: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Logical codewords for the (b,g)-PI code in the top block (dim N+1) Dicke basis.

    Returns: (ket0_top, ket1_top, N) where N = 2b+g
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
    return ket0, ket1, N


def bgcode_piqs(b: int, g: int, *, return_qutip: bool = True):
    """
    PIQS reduced-space embedding for (b,g)-PI code:
      returns rho, l0, l1 in reduced space.
    """
    ket0_top, ket1_top, N = bgcode_kets_in_top_block(b, g)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)


# ============================================================
# (b,g,m)-PI code
# ============================================================

def bgm_gamma_vectors(b: int, g: int, m: int) -> np.ndarray:
    """
    gamma[k] = γ(b,g,m,k) for k=0..m
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


def bgmcode_kets_in_top_block(b: int, g: int, m: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Logical codewords for the (b,g,m)-PI code in the top block (dim N+1) Dicke basis.

    Returns: (ket0_top, ket1_top, N) where N = 2*b*m + g
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
    return ket0, ket1, N


def bgmcode_piqs(b: int, g: int, m: int, *, return_qutip: bool = True):
    """
    PIQS reduced-space embedding for (b,g,m)-PI code:
      returns rho, l0, l1 in reduced space.
    """
    ket0_top, ket1_top, N = bgmcode_kets_in_top_block(b, g, m)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)


# ============================================================
# Gross 13-qubit code (top block dim 14, reduced dim 56)
# ============================================================

def gross_13_kets_in_top_block(phi: float = 0.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Gross code for N=13 qubits in top block (dim 14).
    Returns (ket0_top, ket1_top, N=13).
    """
    N = 13
    dim = N + 1

    c1 = math.sqrt(910) / 56
    c2 = -3 * math.sqrt(154) / 56
    c3 = -math.sqrt(770) / 56
    c4 = math.sqrt(70) / 56

    c5 = math.sqrt(231) / 84
    c6 = math.sqrt(1365) / 84
    c7 = -math.sqrt(273) / 28
    c8 = -math.sqrt(3003) / 84

    basis1 = np.zeros(dim, dtype=complex); basis1[13] = 1.0
    basis2 = np.zeros(dim, dtype=complex); basis2[9]  = 1.0
    basis3 = np.zeros(dim, dtype=complex); basis3[5]  = 1.0
    basis4 = np.zeros(dim, dtype=complex); basis4[1]  = 1.0

    state1 = c1*basis1 + c2*basis2 + c3*basis3 + c4*basis4
    state2 = c5*basis1 + c6*basis2 + c7*basis3 + c8*basis4

    phase = np.exp(1j * phi)
    a = math.sqrt(105) / 14
    b = math.sqrt(91) / 14

    ket0_top = a * state1 + b * phase * state2
    ket1_top = b * state1 - a * np.conjugate(phase) * state2

    ket0_top = ket0_top / np.linalg.norm(ket0_top)
    ket1_top = ket1_top / np.linalg.norm(ket1_top)
    return ket0_top, ket1_top, N


def gross_13_piqs(phi: float = 0.0, *, return_qutip: bool = True):
    """
    PIQS reduced-space embedding for Gross 13-qubit code:
      N=13, blocks [14,12,10,8,6,4,2] total 56.
    """
    ket0_top, ket1_top, N = gross_13_kets_in_top_block(phi)
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)


# ============================================================
# Gross 17-qubit code (top block dim 18, reduced dim 90)
# ============================================================

def gross_17_kets_in_top_block() -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Gross code for N=17 qubits in top block (dim 18).
    Returns (ket0_top, ket1_top, N=17).
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
    PIQS reduced-space embedding for Gross 17-qubit code:
      N=17, blocks [18,16,14,12,10,8,6,4,2] total 90.
    """
    ket0_top, ket1_top, N = gross_17_kets_in_top_block()
    return symmetric_code_to_piqs_reduced(ket0_top, ket1_top, N, return_qutip=return_qutip)


