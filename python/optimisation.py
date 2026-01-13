import numpy as np
import qutip
import cvxpy as cp
from typing import Union, Optional, List, Dict, Tuple


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


def optimise(
    logical0: qutip.Qobj,
    logical1: qutip.Qobj,
    noise: Union[List[qutip.Qobj], qutip.Qobj],
    solver: Union[str, object] = "mosek",
    rho_L: Optional[np.ndarray] = None,
    warm_start: bool = True,
    solver_opts: Optional[dict] = None,
) -> float:
    """
    Optimise recovery via SDP with CVXPY problem reuse (Parameterized objective).

    `noise` can be:
      - list[Qobj] : Kraus operators for d->d
      - Qobj       : Choi matrix for d->d

    Returns optimal objective value (float).
    """
    # --- solver handling ---
    if isinstance(solver, str):
        solver_name = solver.strip().lower()
        solver_map = {
            "scs": cp.SCS,
            "mosek": cp.MOSEK,
        }
        solver = solver_map.get(solver_name, None)
        if solver is None:
            raise ValueError(f"Unsupported solver string '{solver}'. Use 'scs' or 'mosek'.")

    if solver_opts is None:
        solver_opts = {}

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
