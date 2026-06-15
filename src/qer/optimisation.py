"""SDP-based recovery optimization helpers.

This module contains the CVXPY semidefinite program used to optimize a
recovery channel for a two-dimensional logical code. The default solver is the
free SCS solver; MOSEK can be selected explicitly when it is installed and
licensed on the user's machine.
"""

import cvxpy as cp
import numpy as np
import qutip
from typing import Dict, List, Optional, Tuple, Union


def partial_trace_out_choi(Xmat, d_out, d_in):
    """
    Trace out the output system of a Choi matrix expression.

    Parameters
    ----------
    Xmat : array-like or cvxpy.Expression
        Choi matrix or CVXPY matrix expression with block size ``d_in``.
    d_out : int
        Output Hilbert-space dimension.
    d_in : int
        Input Hilbert-space dimension.

    Returns
    -------
    array-like or cvxpy.Expression
        Partial trace over the output system, with shape ``(d_in, d_in)``.
    """
    return sum(
        Xmat[a * d_in:(a + 1) * d_in, a * d_in:(a + 1) * d_in]
        for a in range(d_out)
    )


# Cache: key = d (input dimension), value = (prob, Cparam, X)
_SDP_CACHE: Dict[int, Tuple[cp.Problem, cp.Parameter, cp.Variable]] = {}


def _get_cached_sdp(d: int) -> Tuple[cp.Problem, cp.Parameter, cp.Variable]:
    """
    Return a cached recovery SDP for one physical dimension.

    Parameters
    ----------
    d : int
        Physical Hilbert-space dimension.

    Returns
    -------
    tuple[cvxpy.Problem, cvxpy.Parameter, cvxpy.Variable]
        CVXPY problem, objective matrix parameter, and Choi variable.
    """
    if d in _SDP_CACHE:
        return _SDP_CACHE[d]

    d_out = 2
    n = 2 * d

    Cparam = cp.Parameter((n, n), complex=True)
    X = cp.Variable((n, n), hermitian=True)

    constraints = [
        X >> 0,
        partial_trace_out_choi(X, d_out, d) == np.eye(d),
    ]

    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(X @ Cparam))), constraints)
    _SDP_CACHE[d] = (prob, Cparam, X)
    return prob, Cparam, X


def _normalise_solver(solver: Union[str, object, None]) -> object | None:
    """
    Convert a public solver alias into a CVXPY solver identifier.

    Parameters
    ----------
    solver : str or object or None
        Solver alias such as ``"scs"``, ``"clarabel"``, or ``"mosek"``;
        a CVXPY solver object; or ``None`` to let CVXPY choose.

    Returns
    -------
    object or None
        CVXPY solver identifier, original solver object, or ``None``.
    """
    if solver is None:
        return None
    if not isinstance(solver, str):
        return solver

    solver_name = solver.strip().lower()
    solver_map = {
        "scs": cp.SCS,
        "clarabel": getattr(cp, "CLARABEL", "CLARABEL"),
        "mosek": cp.MOSEK,
    }
    if solver_name not in solver_map:
        supported = ", ".join(sorted(solver_map))
        raise ValueError(f"Unsupported solver string '{solver}'. Use one of: {supported}.")
    return solver_map[solver_name]


def _solver_name(solver: object | None) -> str | None:
    """
    Return a normalized display name for a CVXPY solver identifier.

    Parameters
    ----------
    solver : object or None
        CVXPY solver identifier.

    Returns
    -------
    str or None
        Uppercase solver name, or ``None`` when no solver is specified.
    """
    if solver is None:
        return None
    return str(solver).upper()


def _solver_options(solver: object | None, solver_opts: Optional[dict]) -> dict:
    """
    Merge default solver options with user-provided options.

    Parameters
    ----------
    solver : object or None
        CVXPY solver identifier.
    solver_opts : dict or None
        User-provided solver keyword arguments.

    Returns
    -------
    dict
        Solver options passed to ``Problem.solve``.
    """
    opts = dict(solver_opts or {})
    if _solver_name(solver) == "SCS":
        opts.setdefault("eps", 1e-7)
    return opts


def optimise(
    logical0: qutip.Qobj,
    logical1: qutip.Qobj,
    noise: Union[List[qutip.Qobj], qutip.Qobj],
    solver: Union[str, object, None] = "scs",
    rho_L: Optional[np.ndarray] = None,
    warm_start: bool = True,
    solver_opts: Optional[dict] = None,
) -> float:
    """
    Optimize the best recovery fidelity for a two-dimensional logical code.

    The default solver is SCS, which is free and installed through CVXPY in many
    environments. Pass ``solver="mosek"`` to opt into MOSEK when it is installed
    and licensed locally.

    Parameters
    ----------
    logical0, logical1 : qutip.Qobj
        Logical code kets embedded in the physical Hilbert space.
    noise : list[qutip.Qobj] or qutip.Qobj
        Noise channel as Kraus operators or as a Choi matrix.
    solver : str or object or None, default="scs"
        CVXPY solver alias or solver identifier.
    rho_L : numpy.ndarray or None, optional
        Two-by-two logical input density matrix. If omitted, the maximally
        mixed logical state is used.
    warm_start : bool, default=True
        Whether to warm-start the cached CVXPY solve.
    solver_opts : dict or None, optional
        Extra keyword arguments passed to ``Problem.solve``.

    Returns
    -------
    float
        Optimal SDP objective value.
    """
    solver = _normalise_solver(solver)
    solver_opts = _solver_options(solver, solver_opts)

    # --- dimensions and encoder ---
    d = logical0.shape[0]
    ket0 = qutip.basis(2, 0)
    ket1 = qutip.basis(2, 1)
    Uc = logical0 * ket0.dag() + logical1 * ket1.dag()  # (d x 2)

    if rho_L is None:
        rho_L = np.eye(2) / 2
    rho_L = np.asarray(rho_L, dtype=np.complex128)
    if rho_L.shape != (2, 2):
        raise ValueError(f"rho_L must be shape (2,2), got {rho_L.shape}")

    # --- build C matrix ---
    n = 2 * d

    if isinstance(noise, list):
        # Kraus path
        C = np.zeros((n, n), dtype=np.complex128)
        for K in noise:
            if not isinstance(K, qutip.Qobj):
                raise TypeError("Kraus list must contain only qutip.Qobj operators.")
            K1 = (K * Uc).full()                    # (d x 2)
            temp = (rho_L @ K1.conj().T).T          # (d x 2)
            vec = temp.reshape((n, 1), order="F")
            C += vec @ vec.conj().T

    elif isinstance(noise, qutip.Qobj):
        # Choi path (NumPy-only, matches Kraus method exactly)
        JN = noise
        if JN.shape != (d * d, d * d):
            raise ValueError(f"Choi has shape {JN.shape}, expected {(d*d, d*d)}")

        JN_mat = JN.full()
        Uc_mat = Uc.full()
        Id = np.eye(d)

        M = np.kron(Uc_mat.T, Id)          # (2d x d^2)
        J_comp = M @ JN_mat @ M.conj().T   # (2d x 2d)

        R = np.kron(rho_L, Id)             # (2d x 2d)
        C = R @ np.conjugate(J_comp) @ R.conj().T

    else:
        raise TypeError("noise must be either a list of Kraus Qobj or a Choi Qobj.")

    # --- solve cached SDP ---
    prob, Cparam, X = _get_cached_sdp(d)
    Cparam.value = C

    prob.solve(solver=solver, warm_start=warm_start, **solver_opts)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP did not solve to optimality. Status: {prob.status}")

    return float(prob.value)


def no_recovery(rho, noise):
    """
    Compute entanglement fidelity without applying a recovery channel.

    The noise channel may be supplied as Kraus operators or as a Choi matrix
    using the unnormalized Choi convention.

    Parameters
    ----------
    rho : qutip.Qobj or array-like
        Reference density matrix.
    noise : list[qutip.Qobj] or tuple[qutip.Qobj] or qutip.Qobj
        Noise channel as Kraus operators or as a Choi matrix.

    Returns
    -------
    float
        Entanglement fidelity with no recovery applied.
    """

    # --- Kraus branch ---
    if isinstance(noise, (list, tuple)):
        Fe = 0.0
        for Ek in noise:
            tk = (rho * Ek).tr()          # Tr(rho E_k)
            Fe += abs(tk) ** 2
        return float(np.real_if_close(Fe))

    # --- Choi branch ---
    J = noise
    Jm = np.asarray(J.full(), dtype=np.complex128)
    Jm = (Jm + Jm.conj().T) / 2

    d2 = Jm.shape[0]
    d = int(round(np.sqrt(d2)))
    if d * d != d2 or Jm.shape[1] != d2:
        raise ValueError(f"Choi matrix must be square with dimension d^2 x d^2, got {Jm.shape}")

    w, v = np.linalg.eigh(Jm)
    rho_m = rho.full() if isinstance(rho, qutip.Qobj) else np.asarray(rho, dtype=np.complex128)
    if rho_m.shape != (d, d):
        raise ValueError(f"rho must be shape ({d}, {d}), got {rho_m.shape}")

    Fe = 0.0
    for lam, vec in zip(w, v.T):
        if lam <= 0:
            continue
        K = np.sqrt(lam) * vec.reshape((d, d), order="F")
        tk = np.trace(rho_m @ K)
        Fe += abs(tk) ** 2

    return float(np.real_if_close(Fe))
