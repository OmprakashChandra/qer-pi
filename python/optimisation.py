import numpy as np
import qutip 
from qutip.piqs.piqs import Dicke
import cvxpy as cp
from typing import Union, Optional

def optimize(logical0: qutip.Qobj, logical1: qutip.Qobj, kraus : list[qutip.Qobj], solver: Union[str, object]):

    if isinstance(solver, str):
        solver_name = solver.upper()
        solver_map = { 
            "scs": cp.SCS,
            "mosek" : cp.MOSEK,
        }
    solver = solver_map.get(solver_name, None)
   

    d = logical0.shape[0]
    ket0 = qutip.basis(2, 0)
    ket1 = qutip.basis(2, 1)
    Uc = logical0 * ket0.dag() + logical1 * ket1.dag()  # dims [[d],[2]]

    noise_type = 'local symmetric depolarizing'
    rho_L = np.eye(2) / 2  # maximally mixed logical state
    C = np.zeros((2 * d, 2 * d), dtype=np.complex128)

    for K in kraus:
        K1 = (K * Uc).full()                   # (d, 2)
        temp = (rho_L @ K1.conj().T).T         # (d, 2)
        vec = temp.reshape((2 * d, 1), order="F")
        C += vec @ vec.conj().T

    def partial_trace_out_choi(X, d_out, d_in):
        blocks = []
        for a in range(d_out):
            sl = slice(a * d_in, (a + 1) * d_in)
            blocks.append(X[sl, sl])
        return sum(blocks)

    n = 2 * d
    X = cp.Variable((n, n), hermitian=True)
    constraints = [
        X >> 0,
        partial_trace_out_choi(X, 2, d) == np.eye(d),  # TP constraint
    ]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(X @ C))), constraints)
    prob.solve(solver=solver)  # change if you have another solver installed

    fid_opt = float(prob.value)
    X_opt = X.value

    # --- convert optimal recovery Choi -> Kraus operators R_kraus (map d -> 2) ---
    #choi_R = qutip.Qobj(X_opt, dims=[[2, d], [2, d]])
    #R_kraus = kraus_from_choi_psd(choi_R, d_out=2, d_in=d, tol=1e-12)

    return fid_opt