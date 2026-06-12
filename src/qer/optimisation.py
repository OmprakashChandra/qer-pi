import cvxpy as cp
import numpy as np
import qutip
from typing import Dict, List, Optional, Tuple, Union


def partial_trace_out_choi(Xmat, d_out, d_in):
    """Sum diagonal blocks to trace out the output system of a Choi matrix."""
    return sum(
        Xmat[a * d_in:(a + 1) * d_in, a * d_in:(a + 1) * d_in]
        for a in range(d_out)
    )


# Cache: key = d (input dimension), value = (prob, Cparam, X)
_SDP_CACHE: Dict[int, Tuple[cp.Problem, cp.Parameter, cp.Variable]] = {}


def _get_cached_sdp(d: int) -> Tuple[cp.Problem, cp.Parameter, cp.Variable]:
    """
    Create and cache the CVXPY problem once per d, with C as a Parameter.
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
    """Return a CVXPY solver identifier from a public string alias."""
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
    if solver is None:
        return None
    return str(solver).upper()


def _solver_options(solver: object | None, solver_opts: Optional[dict]) -> dict:
    """Merge public-friendly defaults with user-provided CVXPY solver options."""
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
    Optimise recovery via SDP with CVXPY problem reuse (Parameterized objective).

    ``noise`` can be:
      - list[Qobj] : Kraus operators for d->d
      - Qobj       : Choi matrix for d->d

    The default solver is SCS, which is free and installed through CVXPY in many
    environments. Pass ``solver="mosek"`` to opt into MOSEK when it is installed
    and licensed locally.

    Returns optimal objective value (float).
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
    Entanglement fidelity with NO recovery.

    If noise is:
      - list / tuple of Qobj  -> interpreted as Kraus operators {E_k}
      - Qobj                 -> interpreted as Choi matrix J (d^2 x d^2)

    Choi convention:
      J = (I ⊗ E)(|Ω⟩⟨Ω|),  |Ω⟩ = Σ_i |i⟩⊗|i⟩  (unnormalized)

    Returns:
      float entanglement fidelity
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
