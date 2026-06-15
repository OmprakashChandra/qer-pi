"""Barnum-Knill and Petz recovery utilities.

This module builds analytic recovery maps for a two-dimensional logical code.
It supports Kraus-based workflows, superoperator/Choi workflows, and helper
functions for evaluating the resulting logical entanglement fidelity.
"""

import numpy as np
import qutip as qt


# -----------------------------
# Numerics helper
# -----------------------------
def _pinv_sqrt_pos(A: qt.Qobj, rcond: float = 1e-12) -> qt.Qobj:
    """
    Return the inverse square root of a PSD operator on its support.

    Parameters
    ----------
    A : qutip.Qobj
        Hermitian positive semidefinite operator.
    rcond : float, default=1e-12
        Relative eigenvalue cutoff used for numerical support.

    Returns
    -------
    qutip.Qobj
        Moore-Penrose inverse square root of ``A``.
    """
    evals, evecs = A.eigenstates()
    evals = np.array(evals, dtype=float)
    max_ev = float(evals.max()) if evals.size else 0.0
    thresh = rcond * max_ev if max_ev > 0 else 0.0

    out = 0 * A
    for lam, v in zip(evals, evecs):
        if lam > thresh:
            out += (1.0 / np.sqrt(lam)) * (v * v.dag())
    return out


def _sqrt_pos(A: qt.Qobj, rcond: float = 1e-12) -> qt.Qobj:
    """
    Return the positive square root of a PSD operator.

    Parameters
    ----------
    A : qutip.Qobj
        Hermitian positive semidefinite operator.
    rcond : float, default=1e-12
        Relative eigenvalue cutoff used for numerical support.

    Returns
    -------
    qutip.Qobj
        Positive operator ``sqrt(A)`` with tiny eigenvalues dropped.
    """
    A = (A + A.dag()) / 2
    evals, evecs = A.eigenstates()
    evals = np.array(evals, dtype=float)
    max_ev = float(evals.max()) if evals.size else 0.0
    thresh = rcond * max_ev if max_ev > 0 else 0.0

    out = 0 * A
    for lam, v in zip(evals, evecs):
        if lam > thresh:
            out += np.sqrt(lam) * (v * v.dag())
        elif lam < -thresh:
            raise ValueError(
                f"Expected a positive semidefinite operator, got eigenvalue {lam:.3e}."
            )
    return out


# -----------------------------
# Code helpers
# -----------------------------
def code_isometry(ket0: qt.Qobj, ket1: qt.Qobj) -> qt.Qobj:
    """
    Build the physical-to-logical code isometry matrix.

    Parameters
    ----------
    ket0, ket1 : qutip.Qobj
        Logical code kets in the physical Hilbert space.

    Returns
    -------
    qutip.Qobj
        Matrix ``V`` with columns ``ket0`` and ``ket1``.
    """
    V = qt.Qobj(
        np.hstack([ket0.full(), ket1.full()]),
        dims=[ket0.dims[0], [2]]
    )
    return V

def code_projector(ket0: qt.Qobj, ket1: qt.Qobj) -> qt.Qobj:
    """
    Return the projector onto the logical codespace.

    Parameters
    ----------
    ket0, ket1 : qutip.Qobj
        Logical code kets in the physical Hilbert space.

    Returns
    -------
    qutip.Qobj
        Projector ``|0_L><0_L| + |1_L><1_L|``.
    """
    return ket0 * ket0.dag() + ket1 * ket1.dag()


# -----------------------------
# BK recovery: Kraus method
# -----------------------------
def petz_recovery_kraus(E_kraus, rho: qt.Qobj, rcond: float = 1e-12):
    """
    Build Petz recovery Kraus operators for a reference state.

    For a noise channel E(X) = sum_i E_i X E_i^dag, this implements
        R_i = rho^{1/2} E_i^dag E(rho)^{-1/2}.

    Parameters
    ----------
    E_kraus : list[qutip.Qobj]
        Noise-channel Kraus operators.
    rho : qutip.Qobj
        Reference density matrix or ket used in the Petz map.
    rcond : float, default=1e-12
        Relative eigenvalue cutoff used in inverse square roots.

    Returns
    -------
    list[qutip.Qobj]
        Petz recovery Kraus operators.
    """
    if rho.isket:
        rho = qt.ket2dm(rho)

    rho_sqrt = _sqrt_pos(rho, rcond=rcond)

    omega = 0 * rho
    for E in E_kraus:
        omega += E * rho * E.dag()

    omega_inv_sqrt = _pinv_sqrt_pos(omega, rcond=rcond)
    return [rho_sqrt * E.dag() * omega_inv_sqrt for E in E_kraus]


def bk_recovery_kraus(E_kraus, ket0: qt.Qobj, ket1: qt.Qobj, rcond: float = 1e-12):
    """
    Build Barnum-Knill recovery Kraus operators for a two-dimensional code.

    Parameters
    ----------
    E_kraus : list[qutip.Qobj]
        Physical noise-channel Kraus operators.
    ket0, ket1 : qutip.Qobj
        Logical code kets spanning the codespace.
    rcond : float, default=1e-12
        Relative eigenvalue cutoff used in inverse square roots.

    Returns
    -------
    list[qutip.Qobj]
        Barnum-Knill recovery Kraus operators.
    """
    P = code_projector(ket0, ket1)
    return petz_recovery_kraus(E_kraus, P, rcond=rcond)


def approx_global_ad_petz_recovery_kraus(
    num_qubits: int,
    gamma: float,
    dt: float,
    ket0: qt.Qobj | None = None,
    ket1: qt.Qobj | None = None,
    rho: qt.Qobj | None = None,
    rcond: float = 1e-12,
):
    """
    Build Petz recovery Kraus operators for approximate global amplitude damping.

    Parameters
    ----------
    num_qubits : int
        Number of physical qubits.
    gamma : float
        Noise-rate parameter.
    dt : float
        Evolution time.
    ket0, ket1 : qutip.Qobj, optional
        Logical code kets. Used to build the maximally mixed reference state
        when ``rho`` is not supplied.
    rho : qutip.Qobj or None, optional
        Reference density matrix or ket used in the Petz map.
    rcond : float, default=1e-12
        Relative eigenvalue cutoff used in inverse square roots.

    Returns
    -------
    list[qutip.Qobj]
        Petz recovery Kraus operators for the approximate global AD channel.
    """
    if rho is None:
        if ket0 is None or ket1 is None:
            raise ValueError("Provide either rho or both ket0 and ket1.")
        rho = code_projector(ket0, ket1) / 2

    try:
        from .noisemodel import noisemodel
    except ImportError:
        from noisemodel import noisemodel

    E_kraus = noisemodel(
        "global symmetric amplitude damping",
        num_qubits,
        gamma,
        dt,
        return_rep="kraus",
        dynamics="approx",
    )
    return petz_recovery_kraus(E_kraus, rho, rcond=rcond)


def first_order_global_ad_petz_recovery_kraus(
    num_qubits: int,
    gamma: float,
    dt: float,
    ket0: qt.Qobj | None = None,
    ket1: qt.Qobj | None = None,
    rho: qt.Qobj | None = None,
    rcond: float = 1e-12,
):
    """
    Build Petz recovery Kraus operators for first-order global AD.

    Parameters
    ----------
    num_qubits : int
        Number of physical qubits.
    gamma : float
        Noise-rate parameter.
    dt : float
        Evolution time.
    ket0, ket1 : qutip.Qobj, optional
        Logical code kets. Used to build the maximally mixed reference state
        when ``rho`` is not supplied.
    rho : qutip.Qobj or None, optional
        Reference density matrix or ket used in the Petz map.
    rcond : float, default=1e-12
        Relative eigenvalue cutoff used in inverse square roots.

    Returns
    -------
    list[qutip.Qobj]
        Petz recovery Kraus operators for the first-order global AD channel.
    """
    if rho is None:
        if ket0 is None or ket1 is None:
            raise ValueError("Provide either rho or both ket0 and ket1.")
        rho = code_projector(ket0, ket1) / 2

    try:
        from .noisemodel import noisemodel
    except ImportError:
        from noisemodel import noisemodel

    E_kraus = noisemodel(
        "global symmetric amplitude damping",
        num_qubits,
        gamma,
        dt,
        return_rep="kraus",
        dynamics="first order AD",
    )
    return petz_recovery_kraus(E_kraus, rho, rcond=rcond)


# -----------------------------
# Logical channel via Kraus compression
# -----------------------------
def logical_kraus_from_physical(R_kraus, E_kraus, V: qt.Qobj, drop_tol: float = 1e-12):
    """
    Compress physical noise and recovery Kraus operators to the logical code.

    Parameters
    ----------
    R_kraus : list[qutip.Qobj]
        Recovery Kraus operators.
    E_kraus : list[qutip.Qobj]
        Noise Kraus operators.
    V : qutip.Qobj
        Code isometry returned by :func:`code_isometry`.
    drop_tol : float, default=1e-12
        Norm threshold below which logical Kraus operators are omitted.

    Returns
    -------
    list[qutip.Qobj]
        Logical Kraus operators ``V.dag() * R_a * E_b * V``.
    """
    Adag = V.dag()
    A_list = []
    for R in R_kraus:
        for E in E_kraus:
            A = Adag * (R * E) * V
            if A.norm() > drop_tol:
                A_list.append(A)
    return A_list

def fidelity_logical_subspace_from_kraus(A_list):
    """
    Return entanglement fidelity for the maximally mixed logical state.

    Parameters
    ----------
    A_list : list[qutip.Qobj]
        Logical Kraus operators acting on the two-dimensional logical space.

    Returns
    -------
    float
        Entanglement fidelity ``0.25 * sum_i |Tr(A_i)|^2``.
    """
    s = 0.0
    for A in A_list:
        s += abs(A.tr())**2
    return float(s) / 4.0


def fidelity_with_recovery_kraus(
    ket0: qt.Qobj,
    ket1: qt.Qobj,
    R_kraus,
    noise,
    method: str = "kraus",
    drop_tol: float = 1e-12,
):
    """
    Entanglement fidelity after applying a fixed recovery Kraus map.

    This is useful when the recovery is the analytic four-Kraus Petz map
    derived for the approximate global-AD channel, but the noise channel being
    tested is either:
      - a Kraus list, or
      - a Choi/superoperator representation of the exact dynamics.

    Parameters
    ----------
    ket0, ket1 : qutip.Qobj
        Logical code kets spanning the codespace.
    R_kraus : list[qutip.Qobj]
        Recovery Kraus operators.
    noise : list[qutip.Qobj] or qutip.Qobj
        Noise channel as Kraus operators, Choi matrix, or superoperator.
    method : str, default="kraus"
        Representation to use: ``"kraus"``, ``"choi"``, or ``"super"``.
    drop_tol : float, default=1e-12
        Norm threshold below which logical Kraus operators are omitted.

    Returns
    -------
    float
        Logical entanglement fidelity after noise and recovery.
    """
    method = method.lower().strip()
    rho_L = qt.qeye(2) / 2

    if method == "kraus":
        V = code_isometry(ket0, ket1)
        A_list = logical_kraus_from_physical(R_kraus, noise, V, drop_tol=drop_tol)
        return fidelity_logical_subspace_from_kraus(A_list)

    if method in ("choi", "super"):
        R_super = qt.kraus_to_super(R_kraus)
        E_super = qt.to_super(noise)
        Lambda_super = R_super * E_super
        J_L = logical_choi_from_physical_super(Lambda_super, ket0, ket1)
        return entanglement_fidelity_from_choi(J_L, rho_L)

    raise ValueError("method must be 'kraus', 'choi', or 'super'")


# -----------------------------
# Superoperator utilities
# -----------------------------
def apply_super(S: qt.Qobj, X: qt.Qobj) -> qt.Qobj:
    """
    Apply a superoperator to an operator or state.

    Parameters
    ----------
    S : qutip.Qobj
        Superoperator or channel representation convertible with
        ``qutip.to_super``.
    X : qutip.Qobj
        Operator, density matrix, or ket to evolve.

    Returns
    -------
    qutip.Qobj
        Operator obtained from applying ``S`` to ``X``.
    """
    if X.isket:
        X = qt.ket2dm(X)
    S = qt.to_super(S) if not S.issuper else S

    # Use dense multiplication to avoid dims strictness pitfalls
    v = qt.operator_to_vector(X).full()        # (d^2, 1)
    y = S.full() @ v                          # (d^2, 1)
    yq = qt.Qobj(y, dims=qt.operator_to_vector(X).dims)
    return qt.vector_to_operator(yq)


def projector_super(P: qt.Qobj) -> qt.Qobj:
    """
    Return the projection superoperator for a projector.

    Parameters
    ----------
    P : qutip.Qobj
        Projector or operator used on both sides of the input.

    Returns
    -------
    qutip.Qobj
        Superoperator implementing ``X -> P X P``.
    """
    return qt.spre(P) * qt.spost(P)


# -----------------------------
# BK recovery: Choi/Super method (Kraus-free BK definition)
# -----------------------------
def bk_recovery_super_from_channel(
    channel: qt.Qobj,
    ket0: qt.Qobj,
    ket1: qt.Qobj,
    rcond: float = 1e-12
) -> qt.Qobj:
    """
    Build Barnum-Knill recovery as a superoperator.

    This Kraus-free construction uses the channel and code projector:
    ``R_BK = Proj o E^dag o (X -> Omega^{-1/2} X Omega^{-1/2})``.

    Parameters
    ----------
    channel : qutip.Qobj
        Noise channel as a Choi matrix or superoperator.
    ket0, ket1 : qutip.Qobj
        Logical code kets spanning the codespace.
    rcond : float, default=1e-12
        Relative eigenvalue cutoff used in inverse square roots.

    Returns
    -------
    qutip.Qobj
        Barnum-Knill recovery superoperator.
    """
    P = code_projector(ket0, ket1)

    E_super = qt.to_super(channel)
    E_adj = E_super.dag()

    Omega = apply_super(E_super, P)
    Omega_inv_sqrt = _pinv_sqrt_pos(Omega, rcond=rcond)

    W_super = qt.spre(Omega_inv_sqrt) * qt.spost(Omega_inv_sqrt)
    Proj = projector_super(P)

    return Proj * E_adj * W_super


# -----------------------------
# Logical Choi from physical superoperator (robust)
# -----------------------------
def logical_choi_from_physical_super(
    S_phys: qt.Qobj,
    ket0: qt.Qobj,
    ket1: qt.Qobj
) -> qt.Qobj:
    """
    Construct the logical Choi matrix from a physical superoperator.

    The logical channel is defined by
    ``Lambda_L(rho) = V.dag() * S_phys(V rho V.dag()) * V``.

    Parameters
    ----------
    S_phys : qutip.Qobj
        Physical superoperator acting on the code's Hilbert space.
    ket0, ket1 : qutip.Qobj
        Logical code kets spanning the codespace.

    Returns
    -------
    qutip.Qobj
        Logical Choi matrix with shape ``(4, 4)``.
    """
    V = code_isometry(ket0, ket1)
    Vdag = V.dag()

    # matrix units |m><n|
    e00 = qt.basis(2, 0) * qt.basis(2, 0).dag()
    e01 = qt.basis(2, 0) * qt.basis(2, 1).dag()
    e10 = qt.basis(2, 1) * qt.basis(2, 0).dag()
    e11 = qt.basis(2, 1) * qt.basis(2, 1).dag()
    basis_ops = [e00, e01, e10, e11]

    # Build Choi: sum_{mn} E_mn ⊗ Lambda(E_mn)
    J = 0
    for Emn in basis_ops:
        X_phys = V * Emn * Vdag
        Y_phys = apply_super(S_phys, X_phys)
        Y_L = Vdag * Y_phys * V
        J += qt.tensor(Emn, Y_L)

    return J


# -----------------------------
# Entanglement fidelity from Choi (robust, dense)
# -----------------------------
def entanglement_fidelity_from_choi(J: qt.Qobj, rho: qt.Qobj) -> float:
    """
    Compute entanglement fidelity from a Choi matrix.

    Parameters
    ----------
    J : qutip.Qobj
        Choi matrix or superoperator for the channel.
    rho : qutip.Qobj
        Reference state or density matrix.

    Returns
    -------
    float
        Entanglement fidelity ``<vec(rho)| J |vec(rho)>`` using
        column-stacking.
    """
    if rho.isket:
        rho = qt.ket2dm(rho)

    # If user accidentally passes a superoperator, convert to choi
    Jc = qt.to_choi(J) if J.issuper else J

    v = qt.operator_to_vector(rho).full()      # (d^2, 1)
    Jm = Jc.full()                             # (d^2, d^2)

    if Jm.shape[0] != v.shape[0]:
        raise ValueError(f"Choi dimension mismatch: J is {Jm.shape}, vec(rho) is {v.shape}")

    Fe = (v.conj().T @ Jm @ v).item()
    return float(np.real(Fe))


# -----------------------------
# Top-level: fidelity BK recovery with selectable method
# -----------------------------
def fidelity_bk_recovery(
    ket0: qt.Qobj,
    ket1: qt.Qobj,
    noise,
    method: str = "kraus",
    rcond: float = 1e-12,
    kraus_tol: float = 1e-10,
    drop_tol: float = 1e-12,
):
    """
    Compute entanglement fidelity with Barnum-Knill recovery.

    Parameters
    ----------
    ket0, ket1 : qutip.Qobj
        Logical code kets spanning the codespace.
    noise : list[qutip.Qobj] or qutip.Qobj
        Noise channel. Use Kraus operators when ``method="kraus"`` and a Choi
        matrix or superoperator when ``method`` is ``"choi"`` or ``"super"``.
    method : str, default="kraus"
        Representation used for the Barnum-Knill construction.
    rcond : float, default=1e-12
        Relative eigenvalue cutoff used in inverse square roots.
    kraus_tol : float, default=1e-10
        Reserved tolerance for future Kraus conversion paths.
    drop_tol : float, default=1e-12
        Norm threshold below which logical Kraus operators are omitted.

    Returns
    -------
    float
        Logical entanglement fidelity after Barnum-Knill recovery.
    """
    method = method.lower().strip()

    rho_L = qt.qeye(2) / 2

    if method == "kraus":
        E_kraus = noise
        R_kraus = bk_recovery_kraus(E_kraus, ket0, ket1, rcond=rcond)
        V = code_isometry(ket0, ket1)
        A_list = logical_kraus_from_physical(R_kraus, E_kraus, V, drop_tol=drop_tol)
        return fidelity_logical_subspace_from_kraus(A_list)

    elif method in ("choi", "super"):
        # BK recovery (super), no Kraus decomposition of noise needed
        R_super = bk_recovery_super_from_channel(noise, ket0, ket1, rcond=rcond)

        # Noise channel as super
        E_super = qt.to_super(noise)

        # Total physical channel: Lambda = R ∘ E
        Lambda_super = R_super * E_super

        # Logical Choi directly (stable dims)
        J_L = logical_choi_from_physical_super(Lambda_super, ket0, ket1)

        # Fe from Choi
        return entanglement_fidelity_from_choi(J_L, rho_L)

    else:
        raise ValueError("method must be 'kraus', 'choi', or 'super'")
