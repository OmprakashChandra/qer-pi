"""
Dicke-subspace (dimension N+1) time-dependent pure-decoherence channel
and its Choi representation, implemented with QuTiP.

Model:
    dρ/dt = γ(t) * D[J_axis] ρ

Exact channel:
    Φ_t = exp( Γ(t) * D[J_axis] ),
    Γ(t) = ∫_0^t γ(s) ds

Returns only:
  - choi_list: [to_choi(Φ_t)] for each t in times
  - G: Γ(t) values on the grid
"""

import numpy as np
import qutip as qt


def integrated_rate(times, gamma=None, Gamma=None, method="trapz"):
    """
    Compute Γ(t)=∫_0^t γ(s) ds on a discrete time grid, or accept Γ(t) directly.

    Provide exactly one of gamma or Gamma.

    Parameters
    ----------
    times : array-like
        Monotone time grid.
    gamma : callable or array-like, optional
        Instantaneous rate γ(t).
    Gamma : callable or array-like, optional
        Integrated rate Γ(t).
    method : str
        "trapz" (default) or "left".

    Returns
    -------
    G : np.ndarray
        Integrated rate values Γ(t_k) with G[0]=0.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("times must be a 1D array with at least one entry.")
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be monotone nondecreasing.")
    if (gamma is None) == (Gamma is None):
        raise ValueError("Provide exactly one of gamma or Gamma (not both).")

    # Γ(t) provided directly
    if Gamma is not None:
        if callable(Gamma):
            G = np.array([float(Gamma(tt)) for tt in t], dtype=float)
        else:
            G = np.asarray(Gamma, dtype=float)
            if G.shape != t.shape:
                raise ValueError("If Gamma is array-like, it must match times shape.")
        return G - G[0]  # enforce Γ(t0)=0

    # γ(t) provided → integrate
    if callable(gamma):
        g = np.array([float(gamma(tt)) for tt in t], dtype=float)
    else:
        g = np.asarray(gamma, dtype=float)
        if g.shape != t.shape:
            raise ValueError("If gamma is array-like, it must match times shape.")

    G = np.zeros_like(t, dtype=float)
    if t.size == 1:
        return G

    dt = np.diff(t)
    if method == "trapz":
        G[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * dt)
    elif method == "left":
        G[1:] = np.cumsum(g[:-1] * dt)
    else:
        raise ValueError("method must be 'trapz' or 'left'.")
    return G


def choi(N, times, gamma=None, Gamma=None, method="trapz", axis="z"):
    """
    Build Choi matrices for the Dicke-subspace (J=N/2) channel:
        dρ/dt = γ(t) * D[J_axis] ρ

    Requires QuTiP's built-in qutip.to_choi.

    Parameters
    ----------
    N : int
        Number of spins (Dicke dimension d = N+1).
    times : array-like
        Time grid (monotone). Returns Φ(0->t_k) for each t_k.
    gamma : callable or array-like, optional
        Time-dependent rate γ(t).
    Gamma : callable or array-like, optional
        Integrated rate Γ(t). If given, gamma must be None.
    method : str
        Integration method for Γ from γ: "trapz" or "left".
    axis : str
        Collective spin operator: "x", "y", or "z".

    Returns
    -------
    choi_list : list[qutip.Qobj]
        Choi matrices J(t_k) on the Dicke subspace.
    G : np.ndarray
        Γ(t_k) values on the grid.
    """
    if not hasattr(qt, "to_choi"):
        raise RuntimeError(
            "Your QuTiP version does not expose qutip.to_choi. "
            "Upgrade QuTiP or implement a manual super->choi converter."
        )

    G = integrated_rate(times, gamma=gamma, Gamma=Gamma, method=method)

    J = N / 2.0
    if axis == "x":
        L = qt.jmat(J, "x")
    elif axis == "y":
        L = qt.jmat(J, "y")
    elif axis == "z":
        L = qt.jmat(J, "z")
    else:
        raise ValueError("axis must be one of {'x','y','z'}.")

    L0 = qt.lindblad_dissipator(L)

    choi_list = []
    for Gt in G:
        Phi_t = (Gt * L0).expm()
        choi_list.append(qt.to_choi(Phi_t))

    return choi_list, G
