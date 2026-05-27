"""
Geometric-phase gate (GPG) utilities in the symmetric Dicke basis.

This module ports the core MATLAB pulse convention used in
``state_prepe_gpg.m`` and ``value_err_H.m``:

    per pulse: U_n = Rz(alpha_n) Ry(beta_n) Rz(gamma_n) G(kappa_n)

where the state is stored in Dicke-weight order ``|D_0^N>, ..., |D_N^N>`` and
the corresponding spin labels are ``m = -N/2, ..., N/2``. The default
reference state is therefore ``|D_0^N>`` / ``m=-N/2``, matching the MATLAB
``Psi(1)=1`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import qutip as qt


ArrayLike = Any


@dataclass
class GPGStatePrepResult:
    """Result returned by :func:`optimize_state_gpg`."""

    xbest: np.ndarray
    X: np.ndarray
    Fbest: float
    fbest: float
    objbest: float
    psi_best: np.ndarray
    psi_target: np.ndarray
    m_values: np.ndarray
    trace: list[dict[str, Any]]
    rho_best: Optional[qt.Qobj] = None
    gpg_mode: str = "noiseless"
    cooperativity: Optional[float] = None
    X_with_detuning: Optional[np.ndarray] = None
    detunings: Optional[np.ndarray] = None


def dicke_dim(num_qubits: int) -> int:
    """Dimension of the symmetric Dicke subspace for ``num_qubits`` qubits."""
    if int(num_qubits) != num_qubits or num_qubits < 0:
        raise ValueError("num_qubits must be a non-negative integer.")
    return int(num_qubits) + 1


def dicke_m_values(num_qubits: int) -> np.ndarray:
    """Return spin labels ``m=-N/2,...,N/2`` in Dicke-weight order."""
    num_qubits = int(num_qubits)
    return np.arange(-num_qubits / 2, num_qubits / 2 + 1, dtype=float)


def dicke_basis_state(num_qubits: int, weight: int = 0) -> qt.Qobj:
    """Return ``|D_weight^N>`` in the symmetric Dicke basis."""
    dim = dicke_dim(num_qubits)
    if weight < 0 or weight >= dim:
        raise ValueError(f"weight must be between 0 and {dim - 1}.")
    return qt.basis(dim, int(weight))


def wrap_angle(theta: ArrayLike) -> np.ndarray:
    """Wrap angles to ``[-pi, pi)``."""
    return (np.asarray(theta, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def coerce_pulse_params(params: ArrayLike) -> np.ndarray:
    """
    Return pulse parameters as a ``(P, 4)`` array.

    Columns are ``[alpha, beta, gamma, kappa]``.
    """
    arr = np.asarray(params, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 4)
    if arr.ndim == 1:
        if arr.size % 4 != 0:
            raise ValueError("Flat pulse parameter arrays must have length 4*P.")
        return arr.reshape(-1, 4)
    if arr.ndim == 2 and arr.shape[1] == 4:
        return arr
    raise ValueError("params must be flat length 4*P or a (P, 4) array.")


def coerce_detuned_pulse_params(params: ArrayLike, detuning_seed: float = 1.0) -> np.ndarray:
    """
    Return pulse parameters as a ``(P, 5)`` array.

    Columns are ``[alpha, beta, gamma, kappa, detuning]``.  Existing
    four-parameter GPG sequences are accepted and padded with ``detuning_seed``.
    """
    arr = np.asarray(params, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 5)
    if arr.ndim == 1:
        if arr.size % 5 == 0:
            return arr.reshape(-1, 5)
        if arr.size % 4 == 0:
            x4 = arr.reshape(-1, 4)
            out = np.zeros((x4.shape[0], 5), dtype=float)
            out[:, :4] = x4
            out[:, 4] = detuning_seed
            return out
        raise ValueError("Flat detuned pulse arrays must have length 5*P or 4*P.")
    if arr.ndim == 2 and arr.shape[1] == 5:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 4:
        out = np.zeros((arr.shape[0], 5), dtype=float)
        out[:, :4] = arr
        out[:, 4] = detuning_seed
        return out
    raise ValueError("params must be flat or a (P, 4)/(P, 5) array.")


def pulse_params_without_detuning(params: ArrayLike) -> np.ndarray:
    """Return the coherent ``[alpha,beta,gamma,kappa]`` part of a pulse sequence."""
    arr = np.asarray(params, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 4)
    if arr.ndim == 2 and arr.shape[1] == 5:
        return arr[:, :4]
    return coerce_pulse_params(arr)


def normalize_gpg_mode(gpg_mode: str) -> str:
    """Normalize user-facing GPG mode aliases."""
    mode = str(gpg_mode).strip().lower().replace("_", "-")
    aliases = {
        "ideal": "noiseless",
        "error-free": "noiseless",
        "errorfree": "noiseless",
        "clean": "noiseless",
        "noisy": "erroneous",
        "error": "erroneous",
        "err": "erroneous",
        "detuned-noisy": "detuned",
        "noisy-detuned": "detuned",
        "detuned-erroneous": "detuned",
        "erroneous-detuned": "detuned",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"noiseless", "erroneous", "detuned"}:
        raise ValueError("gpg_mode must be 'noiseless', 'erroneous', or 'detuned'.")
    return mode


def _validate_gpg_mode_settings(gpg_mode: str, cooperativity: Optional[float]) -> str:
    """Return a normalized GPG mode and check the required noise parameters."""
    mode = normalize_gpg_mode(gpg_mode)
    if mode in {"erroneous", "detuned"}:
        if cooperativity is None or np.isinf(cooperativity):
            raise ValueError(f"{mode} GPG mode requires a finite cooperativity.")
        if cooperativity <= 0:
            raise ValueError("cooperativity must be positive.")
    return mode


def _recorded_gpg_cooperativity(gpg_mode: str, cooperativity: Optional[float]) -> float:
    """Return the effective cooperativity value to store in records."""
    if normalize_gpg_mode(gpg_mode) == "noiseless":
        return float(np.inf)
    if cooperativity is None:
        raise ValueError("erroneous GPG records require cooperativity.")
    return float(cooperativity)


@lru_cache(maxsize=None)
def collective_x_twice(num_qubits: int) -> qt.Qobj:
    """
    Return the MATLAB ``A`` matrix, equal to ``2 J_x`` in the Dicke basis.

    MATLAB builds this by placing ``sqrt(S(S+1)-m(m+1))`` on the nearest
    off-diagonals.
    """
    dim = dicke_dim(num_qubits)
    S = num_qubits / 2
    A = np.zeros((dim, dim), dtype=complex)
    for idx, m in enumerate(np.arange(-S, S, dtype=float)):
        c = np.sqrt(S * (S + 1) - m * (m + 1))
        A[idx, idx + 1] = c
        A[idx + 1, idx] = c
    return qt.Qobj(A, dims=[[dim], [dim]])


@lru_cache(maxsize=None)
def rx_pi2(num_qubits: int) -> qt.Qobj:
    """Return ``exp(i*pi/4 * 2J_x)``, the MATLAB ``Rx_pi2``/``ESx`` helper."""
    return (1j * np.pi / 4 * collective_x_twice(num_qubits)).expm()


def rz(num_qubits: int, theta: float) -> qt.Qobj:
    """Collective ``Rz(theta) = diag(exp(-i theta m))``."""
    m_values = dicke_m_values(num_qubits)
    return qt.Qobj(
        np.diag(np.exp(-1j * theta * m_values)),
        dims=[[len(m_values)], [len(m_values)]],
    )


def ry(num_qubits: int, theta: float) -> qt.Qobj:
    """Collective ``Ry(theta)`` using the MATLAB ``Rx_pi2 * Rz * Rx_pi2'`` convention."""
    R = rx_pi2(num_qubits)
    return R * rz(num_qubits, theta) * R.dag()


def gpg_phase(num_qubits: int, kappa: float) -> qt.Qobj:
    """Nonlinear GPG phase ``G(kappa) = diag(exp(-i kappa m^2))``."""
    m_values = dicke_m_values(num_qubits)
    return qt.Qobj(
        np.diag(np.exp(-1j * kappa * (m_values**2))),
        dims=[[len(m_values)], [len(m_values)]],
    )


def rotation_zyz(num_qubits: int, alpha: float, beta: float, gamma: float) -> qt.Qobj:
    """Return ``Rz(alpha) Ry(beta) Rz(gamma)``."""
    return rz(num_qubits, alpha) * ry(num_qubits, beta) * rz(num_qubits, gamma)


def gpg_pulse(
    num_qubits: int,
    alpha: float,
    beta: float,
    gamma: float,
    kappa: float,
    *,
    wrap: bool = True,
) -> qt.Qobj:
    """Return one MATLAB-style pulse matrix."""
    if wrap:
        alpha, beta, gamma, kappa = wrap_angle([alpha, beta, gamma, kappa])
    return rotation_zyz(num_qubits, alpha, beta, gamma) * gpg_phase(num_qubits, kappa)


def gpg_sequence_unitary(
    num_qubits: int,
    params: ArrayLike,
    *,
    wrap: bool = True,
) -> qt.Qobj:
    """Return the unitary generated by a GPG pulse sequence."""
    pulses = coerce_pulse_params(params)
    U = qt.qeye(dicke_dim(num_qubits))
    for alpha, beta, gamma, kappa in pulses:
        U = gpg_pulse(
            num_qubits,
            alpha,
            beta,
            gamma,
            kappa,
            wrap=wrap,
        ) * U
    return U


def apply_gpg_sequence(
    num_qubits: int,
    params: ArrayLike,
    state: Optional[ArrayLike] = None,
    *,
    normalize: bool = True,
    wrap: bool = True,
) -> qt.Qobj:
    """Apply a noiseless GPG sequence to a state ket."""
    if state is None:
        ket = dicke_basis_state(num_qubits, 0)
    elif isinstance(state, qt.Qobj):
        ket = state
    else:
        arr = np.asarray(state, dtype=complex).reshape(-1, 1)
        ket = qt.Qobj(arr, dims=[[dicke_dim(num_qubits)], [1]])

    out = gpg_sequence_unitary(num_qubits, params, wrap=wrap) * ket
    if normalize:
        nrm = out.norm()
        if nrm > 0:
            out = out / nrm
    return out


def gpg_dephasing_factor(
    num_qubits: int,
    kappa: float,
    cooperativity: Optional[float],
) -> np.ndarray:
    """
    Multiplicative density-matrix error factor from ``value_err_H.m``.

    ``cooperativity=np.inf`` returns the identity factor.
    """
    dim = dicke_dim(num_qubits)
    if cooperativity is None or np.isinf(cooperativity):
        return np.ones((dim, dim), dtype=float)
    if cooperativity <= 0:
        raise ValueError("cooperativity must be positive.")

    indices = np.arange(dim)
    row_indices, col_indices = np.meshgrid(indices, indices)
    coop = np.sqrt(cooperativity)
    cos_num = np.sqrt(2 * (1 + 2 ** (-num_qubits)))
    pulse_area = abs(float(kappa))

    return np.exp(
        -((col_indices - row_indices) ** 2) * pulse_area * cos_num / (2 * coop)
        - (col_indices + row_indices) * pulse_area / (2 * cos_num * coop)
    )


def detuned_gpg_dephasing_factor(
    num_qubits: int,
    kappa: float,
    detuning: float,
    cooperativity: Optional[float],
) -> np.ndarray:
    """
    Multiplicative density-matrix factor for the detuned noisy GPG map.

    This keeps the usual MATLAB coherent gate ``exp(-i kappa J_z^2)`` and
    replaces the finite-cooperativity loss factor by the detuned form
    proportional to ``(n-m)^2/delta + (n+m) delta`` in Dicke-weight order.
    """
    dim = dicke_dim(num_qubits)
    if cooperativity is None or np.isinf(cooperativity):
        return np.ones((dim, dim), dtype=float)
    if cooperativity <= 0:
        raise ValueError("cooperativity must be positive.")

    delta = abs(float(detuning))
    if delta <= 0:
        raise ValueError("detuning must be nonzero.")

    indices = np.arange(dim, dtype=float)
    row_indices, col_indices = np.meshgrid(indices, indices)
    pulse_area = abs(float(kappa))
    loss = ((col_indices - row_indices) ** 2) / delta + (col_indices + row_indices) * delta
    return np.exp(-pulse_area * loss / (2 * np.sqrt(cooperativity)))


def _density_array_and_dims(num_qubits: int, rho_or_state: Optional[ArrayLike]):
    dim = dicke_dim(num_qubits)
    if rho_or_state is None:
        ket = dicke_basis_state(num_qubits, 0)
        rho = qt.ket2dm(ket)
        return rho.full(), rho.dims, True

    if isinstance(rho_or_state, qt.Qobj):
        rho = qt.ket2dm(rho_or_state) if rho_or_state.isket else rho_or_state
        return rho.full(), rho.dims, True

    arr = np.asarray(rho_or_state, dtype=complex)
    if arr.ndim == 1 or arr.shape == (dim, 1) or arr.shape == (1, dim):
        ket = arr.reshape(dim, 1)
        rho = ket @ ket.conj().T
    elif arr.shape == (dim, dim):
        rho = arr
    else:
        raise ValueError(f"rho_or_state must be a ket of length {dim} or a {dim}x{dim} matrix.")
    return rho, [[dim], [dim]], False


def apply_gpg_sequence_density(
    num_qubits: int,
    params: ArrayLike,
    rho_or_state: Optional[ArrayLike] = None,
    *,
    cooperativity: Optional[float] = None,
    wrap: bool = True,
    return_qobj: Optional[bool] = None,
) -> qt.Qobj | np.ndarray:
    """
    Apply a GPG sequence to a density matrix, optionally including GPG errors.

    If ``cooperativity`` is ``None`` or ``np.inf``, this is the noiseless density
    evolution. Finite cooperativity applies the elementwise error factor used in
    the MATLAB ``value_err_H.m`` routine after each ``G(kappa)`` pulse.
    """
    rho, dims, input_was_qobj = _density_array_and_dims(num_qubits, rho_or_state)
    pulses = coerce_pulse_params(params)
    if return_qobj is None:
        return_qobj = input_was_qobj

    for alpha, beta, gamma, kappa in pulses:
        if wrap:
            alpha, beta, gamma, kappa = wrap_angle([alpha, beta, gamma, kappa])

        G = gpg_phase(num_qubits, kappa).full()
        rho = G @ rho @ G.conj().T
        rho = gpg_dephasing_factor(num_qubits, kappa, cooperativity) * rho

        R = rotation_zyz(num_qubits, alpha, beta, gamma).full()
        rho = R @ rho @ R.conj().T

    rho = (rho + rho.conj().T) / 2
    if return_qobj:
        return qt.Qobj(rho, dims=dims)
    return rho


def apply_detuned_gpg_sequence_density(
    num_qubits: int,
    params: ArrayLike,
    rho_or_state: Optional[ArrayLike] = None,
    *,
    cooperativity: Optional[float] = None,
    wrap: bool = True,
    return_qobj: Optional[bool] = None,
) -> qt.Qobj | np.ndarray:
    """
    Apply a GPG sequence with an optimized detuning per pulse to a density matrix.

    The coherent pulse is the usual ``Rz Ry Rz G(kappa)`` sequence.  The fifth
    parameter only changes the finite-cooperativity loss factor.
    """
    rho, dims, input_was_qobj = _density_array_and_dims(num_qubits, rho_or_state)
    pulses = coerce_detuned_pulse_params(params)
    if return_qobj is None:
        return_qobj = input_was_qobj

    for alpha, beta, gamma, kappa, detuning in pulses:
        if wrap:
            alpha, beta, gamma, kappa = wrap_angle([alpha, beta, gamma, kappa])

        G = gpg_phase(num_qubits, kappa).full()
        rho = G @ rho @ G.conj().T
        rho = detuned_gpg_dephasing_factor(num_qubits, kappa, detuning, cooperativity) * rho

        R = rotation_zyz(num_qubits, alpha, beta, gamma).full()
        rho = R @ rho @ R.conj().T

    rho = (rho + rho.conj().T) / 2
    if return_qobj:
        return qt.Qobj(rho, dims=dims)
    return rho


def state_prep_density_fidelity(
    num_qubits: int,
    params: ArrayLike,
    target_state: ArrayLike,
    *,
    initial_state: Optional[ArrayLike] = None,
    cooperativity: Optional[float] = None,
    wrap: bool = True,
) -> tuple[float, float]:
    """
    Return unnormalized target fidelity and final trace for noisy state prep.

    The erroneous GPG map is trace decreasing.  We therefore use
    ``<target|rho|target>`` without renormalizing ``rho`` so that loss from the
    finite-cooperativity factor is penalized during optimization.
    """
    target = normalize_state(target_state).reshape(-1, 1)
    rho = apply_gpg_sequence_density(
        num_qubits,
        params,
        initial_state,
        cooperativity=cooperativity,
        wrap=wrap,
        return_qobj=False,
    )
    fidelity = float(np.real((target.conj().T @ rho @ target).item()))
    trace = float(np.real(np.trace(rho)))
    return min(1.0, max(0.0, fidelity)), max(0.0, trace)


def detuned_state_prep_density_fidelity(
    num_qubits: int,
    params: ArrayLike,
    target_state: ArrayLike,
    *,
    initial_state: Optional[ArrayLike] = None,
    cooperativity: Optional[float] = None,
    wrap: bool = True,
) -> tuple[float, float]:
    """Return unnormalized target fidelity and trace for detuned noisy state prep."""
    target = normalize_state(target_state).reshape(-1, 1)
    rho = apply_detuned_gpg_sequence_density(
        num_qubits,
        params,
        initial_state,
        cooperativity=cooperativity,
        wrap=wrap,
        return_qobj=False,
    )
    fidelity = float(np.real((target.conj().T @ rho @ target).item()))
    trace = float(np.real(np.trace(rho)))
    return min(1.0, max(0.0, fidelity)), max(0.0, trace)


def state_overlap(a: ArrayLike, b: ArrayLike) -> complex:
    """Return ``<a|b>`` after flattening both state vectors."""
    avec = a.full().ravel() if isinstance(a, qt.Qobj) else np.asarray(a, dtype=complex).ravel()
    bvec = b.full().ravel() if isinstance(b, qt.Qobj) else np.asarray(b, dtype=complex).ravel()
    return np.vdot(avec, bvec)


def normalize_state(state: ArrayLike) -> np.ndarray:
    """Return a normalized dense state vector."""
    vec = state.full().ravel() if isinstance(state, qt.Qobj) else np.asarray(state, dtype=complex).ravel()
    nrm = np.linalg.norm(vec)
    if nrm == 0:
        raise ValueError("state must be nonzero.")
    return vec / nrm


def phase_insensitive_overlap(a: ArrayLike, b: ArrayLike) -> float:
    """Return ``|<a|b>|`` after normalizing both state vectors."""
    return float(abs(np.vdot(normalize_state(a), normalize_state(b))))


def phase_aligned_state_error(target: ArrayLike, achieved: ArrayLike) -> float:
    """Return ``||achieved - exp(i phi) target||`` with the best global phase."""
    target_vec = normalize_state(target)
    achieved_vec = normalize_state(achieved)
    overlap = np.vdot(target_vec, achieved_vec)
    phase = np.exp(1j * np.angle(overlap)) if abs(overlap) > 0 else 1.0
    return float(np.linalg.norm(achieved_vec - phase * target_vec))


def p_cache_key(p: float) -> str:
    """Canonical cache key for sweep points indexed by the AD parameter ``p``."""
    return f"{float(p):.12e}"


def gpg_recovery_cache_key(
    p: float,
    *,
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
) -> str:
    """Canonical sweep-cache key including the GPG noise model."""
    mode = normalize_gpg_mode(gpg_mode)
    base = p_cache_key(p)
    if mode == "noiseless":
        return base
    if cooperativity is None:
        raise ValueError("erroneous GPG cache keys require cooperativity.")
    return f"{base}|gpg={mode}|C={float(cooperativity):.12e}"


def sweep_phase_delta(theta: float, background: float) -> float:
    """Wrap a phase relative to a background phase into ``[-pi, pi)``."""
    return float((float(theta) - float(background) + np.pi) % (2 * np.pi) - np.pi)


def spectral_phase_specs_stable(
    system_unitary: ArrayLike,
    label: str,
    *,
    diagonalizer: Optional[ArrayLike] = None,
    background_theta: float = 0.0,
    phase_tol: float = 1e-9,
) -> list[dict[str, Any]]:
    """
    Return rank-one phase specs using an orthonormal spectral basis.

    Schur decomposition is used when only a unitary is supplied.  This is more
    stable than ``np.linalg.eig`` for nearly degenerate unitary spectra and
    avoids non-orthogonal eigenvectors in the rank-one phase product.
    """
    from scipy.linalg import schur

    U = system_unitary.full() if isinstance(system_unitary, qt.Qobj) else np.asarray(system_unitary)
    U = np.asarray(U, dtype=complex)
    dim = U.shape[0]
    if U.shape != (dim, dim):
        raise ValueError("system_unitary must be a square matrix.")

    if diagonalizer is None:
        _, evecs = schur(U, output="complex")
        evals = np.array([evecs[:, ell].conjugate() @ U @ evecs[:, ell] for ell in range(dim)])
    else:
        H = diagonalizer.full() if isinstance(diagonalizer, qt.Qobj) else np.asarray(diagonalizer)
        H = np.asarray(H, dtype=complex)
        H = (H + H.conjugate().T) / 2
        _, evecs = np.linalg.eigh(H)
        evals = np.array([evecs[:, ell].conjugate() @ U @ evecs[:, ell] for ell in range(dim)])

    specs = []
    for ell in range(dim):
        lam = complex(evals[ell])
        theta = float(np.angle(lam))
        delta = sweep_phase_delta(theta, background_theta)
        if abs(np.exp(1j * delta) - 1.0) <= phase_tol:
            continue
        target = qt.Qobj(evecs[:, ell].reshape(-1, 1), dims=[[dim], [1]])
        specs.append(
            {
                "factor": label,
                "eig_index": int(ell),
                "target": target,
                "theta_l": theta,
                "background theta": float(background_theta),
                "delta_l": delta,
                "|lambda_l|-1": abs(abs(lam) - 1.0),
            }
        )
    return specs


def codespace_mixed_fidelity(rho: qt.Qobj, ket0: qt.Qobj, ket1: qt.Qobj) -> float:
    """Squared Uhlmann fidelity of ``rho`` to ``I/2`` on ``span{ket0, ket1}``."""
    logical_block = np.array(
        [
            [ket0.dag() * rho * ket0, ket0.dag() * rho * ket1],
            [ket1.dag() * rho * ket0, ket1.dag() * rho * ket1],
        ],
        dtype=complex,
    )
    logical_block = (logical_block + logical_block.conjugate().T) / 2
    evals = np.linalg.eigvalsh(logical_block)
    evals = np.clip(evals, 0.0, None)
    fid = float((np.sqrt(evals).sum() ** 2) / 2)
    return min(1.0, max(0.0, fid))


def state_prep_sequence_record(
    num_qubits: int,
    target: ArrayLike,
    params: ArrayLike,
    *,
    initial_state: Optional[ArrayLike] = None,
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
    source: str = "cached",
    elapsed_s: float = 0.0,
    restarts: int = 0,
    maxiter: int = 0,
) -> dict[str, Any]:
    """Return the standard bookkeeping row for an existing GPG state-prep sequence."""
    mode = _validate_gpg_mode_settings(gpg_mode, cooperativity)
    pulses_detuned = None
    if mode == "detuned":
        pulses_detuned = coerce_detuned_pulse_params(params)
        pulses = pulses_detuned[:, :4]
    else:
        pulses = coerce_pulse_params(params)
    if mode == "noiseless":
        achieved = apply_gpg_sequence(num_qubits, pulses, initial_state)
        overlap = phase_insensitive_overlap(target, achieved)
        fidelity = float(overlap**2)
        trace = 1.0
        phase_error = phase_aligned_state_error(target, achieved)
    elif mode == "erroneous":
        fidelity, trace = state_prep_density_fidelity(
            num_qubits,
            pulses,
            target,
            initial_state=initial_state,
            cooperativity=cooperativity,
        )
        phase_error = np.nan
    else:
        fidelity, trace = detuned_state_prep_density_fidelity(
            num_qubits,
            pulses_detuned,
            target,
            initial_state=initial_state,
            cooperativity=cooperativity,
        )
        phase_error = np.nan

    record = {
        "X": pulses,
        "F_state": fidelity,
        "1 - F_state": float(1 - fidelity),
        "Tr_state": trace,
        "phase-aligned state error": phase_error,
        "gpg_mode": mode,
        "cooperativity": _recorded_gpg_cooperativity(mode, cooperativity),
        "elapsed_s": float(elapsed_s),
        "pulses": len(pulses),
        "restarts": int(restarts),
        "maxiter": int(maxiter),
        "source": source,
    }
    if pulses_detuned is not None:
        record["X_detuned"] = pulses_detuned
        record["detunings"] = pulses_detuned[:, 4].copy()
    return record


def reusable_sequence_for_target(
    target: ArrayLike,
    cache: list[dict[str, Any]],
    *,
    overlap_tol: float = 1e-10,
) -> Optional[dict[str, Any]]:
    """Return a cached sequence whose target matches up to global phase."""
    for item in cache:
        if phase_insensitive_overlap(target, item["target"]) > 1 - overlap_tol:
            return item
    return None


def load_gpg_sweep_cache(path: str | Path) -> dict[str, Any]:
    """Load a pickled GPG sweep cache."""
    import pickle

    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_gpg_sweep_cache(cache: dict[str, Any], path: str | Path) -> None:
    """Write a pickled GPG sweep cache."""
    import pickle

    with Path(path).open("wb") as f:
        pickle.dump(cache, f)


def cache_point_for_p(
    cache: dict[str, Any],
    p: float,
    *,
    cache_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return the cached sweep point for ``p``."""
    key = p_cache_key(p)
    if key not in cache["points"]:
        location = f" in {cache_path}" if cache_path is not None else ""
        raise KeyError(f"p={p} / key={key} is not in the GPG sweep cache{location}")
    return cache["points"][key]


def gpg_sweep_rows(cache: dict[str, Any]):
    """Return a ``p``-sorted DataFrame of sweep metrics."""
    import pandas as pd

    return pd.DataFrame([pt["metrics"] for pt in cache["points"].values()]).sort_values("p")


def default_spike_points(cache: dict[str, Any], min_penalty: float) -> list[float]:
    """Return sweep points whose GPG penalty exceeds ``min_penalty``."""
    df = gpg_sweep_rows(cache)
    mask = df["GPG - exact infidelity penalty"] > min_penalty
    return [float(p) for p in df.loc[mask, "p"]]


def suspect_refinement_keys(
    point: dict[str, Any],
    *,
    state_tol: float,
    synth_tol: float,
) -> set[tuple[str, int]]:
    """Return ``(factor, eig_index)`` targets likely responsible for a spike."""
    keys: set[tuple[str, int]] = set()
    for record in point.get("optimization_records", []):
        if float(record.get("1 - F_state", 0.0)) >= state_tol:
            keys.add((record["factor"], int(record["eig_index"])))

    split_labels = {
        "split 0": "stage 0 split entangler",
        "split 1": "stage 1 split entangler",
        "split 2": "stage 2 split entangler",
    }
    for record in point.get("synthesis_quality", []):
        if float(record.get("||GPG - target||", 0.0)) < synth_tol:
            continue

        factor = record["factor"]
        opt_factor = split_labels.get(factor, factor if factor.startswith("feedback U_") else None)
        if opt_factor is None:
            continue

        for opt in point.get("optimization_records", []):
            if opt["factor"] == opt_factor:
                keys.add((opt["factor"], int(opt["eig_index"])))

    return keys


def refinement_settings(
    label: str,
    eig_index: int,
    old_params: Optional[ArrayLike],
    *,
    mode: str = "normal",
) -> dict[str, int]:
    """Optimization budget used when re-running a suspect GPG target."""
    old_pulses = len(coerce_pulse_params(old_params)) if old_params is not None else 0
    if mode == "quick":
        extra = 1
        restarts = 5
        maxiter = 550
    elif mode == "deep":
        extra = 3
        restarts = 12
        maxiter = 1200
    elif mode == "normal":
        extra = 2
        restarts = 8
        maxiter = 850
    else:
        raise ValueError("mode must be one of: quick, normal, deep.")

    pulses = max(old_pulses + extra, 9)
    if "split entangler" in label:
        pulses = max(pulses, 10)
    if label == "feedback U_2":
        pulses = max(pulses, 10)
    if mode == "deep" and label == "feedback U_2":
        pulses = max(pulses, 11)

    seed = 9000 + 97 * int(eig_index) + 11 * sum(ord(ch) for ch in label)
    return {"pulses": pulses, "restarts": restarts, "maxiter": maxiter, "seed": seed}


def sqrt_pos(A: qt.Qobj, rcond: float = 1e-12, atol: float = 1e-10) -> qt.Qobj:
    """Positive square-root of a Hermitian positive semidefinite operator."""
    A = (A + A.dag()) / 2
    evals, evecs = A.eigenstates()
    evals = np.array(evals, dtype=float)
    max_ev = float(evals.max()) if evals.size else 0.0
    thresh = rcond * max_ev if max_ev > 0 else 0.0
    tol = max(atol, thresh)

    out = 0 * A
    for lam, vec in zip(evals, evecs):
        if lam > thresh:
            out += np.sqrt(lam) * (vec * vec.dag())
        elif lam < -tol:
            raise ValueError(f"Expected a PSD operator, got eigenvalue {lam:.3e}.")
    return out


def pinv_pos(A: qt.Qobj, rcond: float = 1e-12) -> qt.Qobj:
    """Moore-Penrose pseudoinverse of a positive semidefinite operator."""
    A = (A + A.dag()) / 2
    evals, evecs = A.eigenstates()
    evals = np.array(evals, dtype=float)
    max_ev = float(evals.max()) if evals.size else 0.0
    thresh = rcond * max_ev if max_ev > 0 else 0.0

    out = 0 * A
    for lam, vec in zip(evals, evecs):
        if lam > thresh:
            out += (1.0 / lam) * (vec * vec.dag())
    return out


def pinv_sqrt_pos(A: qt.Qobj, rcond: float = 1e-12) -> qt.Qobj:
    """Moore-Penrose inverse square-root of a positive semidefinite operator."""
    A = (A + A.dag()) / 2
    evals, evecs = A.eigenstates()
    evals = np.array(evals, dtype=float)
    max_ev = float(evals.max()) if evals.size else 0.0
    thresh = rcond * max_ev if max_ev > 0 else 0.0

    out = 0 * A
    for lam, vec in zip(evals, evecs):
        if lam > thresh:
            out += (1.0 / np.sqrt(lam)) * (vec * vec.dag())
    return out


def support_projector(A: qt.Qobj, rcond: float = 1e-12) -> qt.Qobj:
    """Projector onto the numerical support of a PSD operator."""
    A = (A + A.dag()) / 2
    evals, evecs = A.eigenstates()
    evals = np.array(evals, dtype=float)
    max_ev = float(evals.max()) if evals.size else 0.0
    thresh = rcond * max_ev if max_ev > 0 else 0.0

    out = 0 * A
    for lam, vec in zip(evals, evecs):
        if lam > thresh:
            out += vec * vec.dag()
    return out


def closest_unitary(U: ArrayLike) -> qt.Qobj:
    """Return the polar/SVD closest unitary to ``U`` in Frobenius norm."""
    dims = U.dims if isinstance(U, qt.Qobj) else None
    arr = U.full() if isinstance(U, qt.Qobj) else np.asarray(U, dtype=complex)
    U_svd, _, Vh_svd = np.linalg.svd(arr, full_matrices=True)
    out = U_svd @ Vh_svd
    if dims is None:
        dim = out.shape[0]
        dims = [[dim], [dim]]
    return qt.Qobj(out, dims=dims)


def unitary_error(U: qt.Qobj) -> float:
    """Return ``||U^dag U - I||``."""
    return float((U.dag() * U - qt.qeye(U.dims[0])).norm())


def polar_unitary_completion(R: qt.Qobj, rcond: float = 1e-12) -> tuple[qt.Qobj, qt.Qobj]:
    """Return the unitary polar completion ``U`` and PSD factor ``B`` for ``R = U B``."""
    U = closest_unitary(R)
    B = sqrt_pos(R.dag() * R, rcond=rcond)
    return U, B


def eq21_unitary_from_B0_S0(B0: qt.Qobj, S0: qt.Qobj) -> qt.Qobj:
    """Return the two-block split unitary ``[[B0, S0], [S0, -B0]]``."""
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    P0 = zero * zero.dag()
    P1 = one * one.dag()
    flip01 = zero * one.dag()
    flip10 = one * zero.dag()
    return (
        qt.tensor(B0, P0)
        + qt.tensor(S0, flip01)
        + qt.tensor(S0, flip10)
        - qt.tensor(B0, P1)
    )


def eq41_from_controlled_entangler(controlled_entangler: qt.Qobj) -> qt.Qobj:
    """Insert a controlled phase into the Eq. 41 single-ancilla compilation."""
    dim = controlled_entangler.dims[0][0]
    ident_S = qt.qeye(dim)
    H_single = qt.Qobj(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), dims=[[2], [2]])
    S_single = qt.Qobj(np.diag([1, 1j]), dims=[[2], [2]])
    Z_single = qt.Qobj(np.diag([1, -1]), dims=[[2], [2]])
    G_a = qt.tensor(ident_S, S_single.dag() * H_single)
    Z_a = qt.tensor(ident_S, Z_single)
    return G_a * controlled_entangler * G_a.dag() * Z_a


def exact_controlled_entangler_from_system_unitary(system_unitary: qt.Qobj) -> qt.Qobj:
    """Return ``U tensor |0><0| + U^dag tensor |1><1|``."""
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    P0 = zero * zero.dag()
    P1 = one * one.dag()
    return qt.tensor(system_unitary, P0) + qt.tensor(system_unitary.dag(), P1)


def controlled_phase_factor_for_eigenvector(
    W_l: qt.Qobj,
    delta_l: float,
    reference_weight: int,
) -> qt.Qobj:
    """Return the rank-one controlled phase ``W_l^dag Phi_l W_l``."""
    dim = W_l.shape[0]
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    P0 = zero * zero.dag()
    P1 = one * one.dag()
    ref = qt.basis(dim, reference_weight)
    P_ref = ref * ref.dag()
    ident_joint = qt.tensor(qt.qeye(dim), qt.qeye(2))
    Phi_l = ident_joint + qt.tensor(
        P_ref,
        (np.exp(1j * delta_l) - 1) * P0 + (np.exp(-1j * delta_l) - 1) * P1,
    )
    W_joint = qt.tensor(W_l, qt.qeye(2))
    return W_joint.dag() * Phi_l * W_joint


def complete_split_unitary(
    first_top: qt.Qobj,
    first_bottom: qt.Qobj,
    rcond: float = 1e-12,
) -> qt.Qobj:
    """Complete a first block column ``[A; C]`` to a unitary on ``S x qubit``."""
    from scipy.linalg import null_space

    gram = first_top.dag() * first_top + first_bottom.dag() * first_bottom
    gram_inv_sqrt = pinv_sqrt_pos(gram, rcond=rcond)
    A = first_top * gram_inv_sqrt
    C = first_bottom * gram_inv_sqrt

    dim = A.shape[0]
    first_column = np.vstack([A.full(), C.full()])
    second_column = null_space(first_column.conjugate().T)
    if second_column.shape[1] != dim:
        raise ValueError("The supplied first block column could not be completed to a unitary.")

    D = qt.Qobj(second_column[:dim, :], dims=A.dims)
    E = qt.Qobj(second_column[dim:, :], dims=A.dims)

    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    return (
        qt.tensor(A, zero * zero.dag())
        + qt.tensor(D, zero * one.dag())
        + qt.tensor(C, one * zero.dag())
        + qt.tensor(E, one * one.dag())
    )


def background_phase_from_diagonalizer(
    system_unitary: qt.Qobj,
    diagonalizer: qt.Qobj,
    inactive_threshold: float = 0.5,
) -> float:
    """Average phase over the inactive eigenspace of ``diagonalizer``."""
    U = system_unitary.full()
    H = diagonalizer.full()
    H = (H + H.conjugate().T) / 2
    evals, evecs = np.linalg.eigh(H)
    inactive_phases = []
    for idx, val in enumerate(evals):
        if val <= inactive_threshold:
            vec = evecs[:, idx]
            inactive_phases.append(np.angle(vec.conjugate() @ U @ vec))
    if not inactive_phases:
        return 0.0
    return float(np.angle(np.mean(np.exp(1j * np.array(inactive_phases)))))


def controlled_entangler_from_gpg_specs(
    num_qubits: int,
    specs: list[dict[str, Any]],
    background_theta: float,
    *,
    reference_weight: Optional[int] = None,
) -> qt.Qobj:
    """Build a controlled entangler from synthesized rank-one GPG specs."""
    if reference_weight is None:
        reference_weight = num_qubits
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    P0 = zero * zero.dag()
    P1 = one * one.dag()
    C = qt.tensor(
        qt.qeye(num_qubits + 1),
        np.exp(1j * background_theta) * P0 + np.exp(-1j * background_theta) * P1,
    )
    for spec in specs:
        A_l = gpg_sequence_unitary(num_qubits, spec["X"])
        W_l = A_l.dag()
        C_l = controlled_phase_factor_for_eigenvector(W_l, spec["delta_l"], reference_weight)
        C = C_l * C
    return C


def system_unitary_from_gpg_specs(
    num_qubits: int,
    specs: list[dict[str, Any]],
    *,
    background_theta: float = 0.0,
    reference_weight: Optional[int] = None,
) -> qt.Qobj:
    """Build a system-only unitary from synthesized rank-one GPG specs."""
    if reference_weight is None:
        reference_weight = num_qubits
    U = np.exp(1j * background_theta) * qt.qeye(num_qubits + 1)
    for spec in specs:
        A_l = gpg_sequence_unitary(num_qubits, spec["X"])
        phase_l = phase_from_prep_unitary(
            A_l,
            spec["delta_l"],
            reference_weight=reference_weight,
            prep_maps_reference_to_target=True,
        )
        U = phase_l * U
    return U


def apply_kraus_channel(kraus_ops: list[qt.Qobj], rho: qt.Qobj) -> qt.Qobj:
    """Apply a Kraus channel to a density matrix."""
    out = 0 * rho
    for K in kraus_ops:
        out += K * rho * K.dag()
    return out


def direct_recovery_state(recovery_ops: list[qt.Qobj], rho: qt.Qobj) -> qt.Qobj:
    """Apply recovery Kraus operators directly."""
    return apply_kraus_channel(recovery_ops, rho)


def top_block_qobj(A: qt.Qobj, dim: int, *, reverse: Optional[bool] = None) -> qt.Qobj:
    """Restrict an operator to the first ``dim`` basis states.

    PIQS reduced-space operators order the highest-spin block oppositely to the
    ``|D_0>,...,|D_N>`` GPG convention.  When ``A`` is larger than ``dim``, the
    default is therefore to reverse the extracted block.
    """
    if A.shape == (dim, dim):
        return A
    if A.shape[0] < dim or A.shape[1] < dim:
        raise ValueError(f"Cannot take a {dim}x{dim} top block from operator with shape {A.shape}.")
    if reverse is None:
        reverse = A.shape[0] > dim
    block = A.full()[:dim, :dim]
    if reverse:
        block = block[::-1, ::-1]
    return qt.Qobj(block, dims=[[dim], [dim]])


def restrict_operators_to_dimension(
    ops: list[qt.Qobj],
    dim: int,
    *,
    reverse: Optional[bool] = None,
) -> list[qt.Qobj]:
    """Restrict a list of operators to the first ``dim`` basis states."""
    return [top_block_qobj(op, dim, reverse=reverse) for op in ops]


def _embed_top_block_rho(
    rho: qt.Qobj,
    ambient_dim: int,
    *,
    reverse: Optional[bool] = None,
) -> qt.Qobj:
    dim = rho.shape[0]
    if reverse is None:
        reverse = ambient_dim > dim
    block = rho.full()
    if reverse:
        block = block[::-1, ::-1]
    arr = np.zeros((ambient_dim, ambient_dim), dtype=complex)
    arr[:dim, :dim] = block
    return qt.Qobj(arr, dims=[[ambient_dim], [ambient_dim]])


def apply_channel_to_state(
    channel,
    rho: qt.Qobj,
    *,
    project_top_block: bool = True,
) -> qt.Qobj:
    """
    Apply a Kraus list, superoperator, or already-computed noisy state to ``rho``.

    If a PIQS reduced-space channel is larger than ``rho``, the state is embedded
    in the top block, evolved, and the top block is returned.  This is intended
    for collective channels whose top Dicke block is invariant.
    """
    if isinstance(channel, list):
        dim = rho.shape[0]
        ambient_dim = channel[0].shape[0]
        if ambient_dim == dim:
            return apply_kraus_channel(channel, rho)
        if not project_top_block:
            raise ValueError(f"Kraus operators have dimension {ambient_dim}, but rho has dimension {dim}.")
        rho_full = _embed_top_block_rho(rho, ambient_dim)
        out_full = apply_kraus_channel(channel, rho_full)
        return top_block_qobj(out_full, dim)

    if isinstance(channel, qt.Qobj) and channel.issuper:
        dim = rho.shape[0]
        ambient_dim = int(round(np.sqrt(channel.shape[0])))
        if ambient_dim == dim:
            return qt.vector_to_operator(channel * qt.operator_to_vector(rho))
        if not project_top_block:
            raise ValueError(
                f"Superoperator acts on dimension {ambient_dim}, but rho has dimension {dim}."
            )
        rho_full = _embed_top_block_rho(rho, ambient_dim)
        out_full = qt.vector_to_operator(channel * qt.operator_to_vector(rho_full))
        return top_block_qobj(out_full, dim)

    if isinstance(channel, qt.Qobj) and channel.shape == rho.shape:
        return channel

    raise TypeError("channel must be a Kraus list, a superoperator Qobj, or a noisy-state Qobj.")


def density_matrix_fidelity(rho: qt.Qobj, sigma: qt.Qobj) -> float:
    """Squared Uhlmann fidelity between two density matrices."""
    if rho.isket:
        rho = qt.ket2dm(rho)
    if sigma.isket:
        sigma = qt.ket2dm(sigma)
    root = sqrt_pos(rho)
    middle = sqrt_pos(root * sigma * root)
    fid = float(abs(middle.tr()) ** 2)
    return min(1.0, max(0.0, fid))


def output_fidelity(
    output: qt.Qobj,
    reference_rho: qt.Qobj,
    *,
    logical_kets: Optional[tuple[qt.Qobj, qt.Qobj]] = None,
    fidelity_fn: Optional[Callable[[qt.Qobj], float]] = None,
) -> float:
    """Evaluate the recovered-state fidelity using a custom, logical, or state fidelity."""
    if fidelity_fn is not None:
        return float(fidelity_fn(output))
    if logical_kets is not None:
        ket0, ket1 = logical_kets
        return codespace_mixed_fidelity(output, ket0, ket1)
    return density_matrix_fidelity(reference_rho, output)


def projector_on_ancilla_bits(system_dim: int, bits: dict[str, int]) -> qt.Qobj:
    """Projector on a branch of the three ancillas ``a,b,c``."""
    ops = [qt.qeye(system_dim)]
    for label in ["a", "b", "c"]:
        bit = bits.get(label)
        if bit is None:
            ops.append(qt.qeye(2))
        else:
            ket = qt.basis(2, int(bit))
            ops.append(ket * ket.dag())
    return qt.tensor(*ops)


def full_identity_sabc(system_dim: int) -> qt.Qobj:
    """Identity on ``S x a x b x c``."""
    return qt.tensor(qt.qeye(system_dim), qt.qeye(2), qt.qeye(2), qt.qeye(2))


def embed_system_qubit_unitary(
    U_Sq: qt.Qobj,
    target_ancilla: str,
    system_dim: int,
) -> qt.Qobj:
    """Embed a system-plus-one-ancilla unitary into ``S x a x b x c``."""
    labels = ["S", target_ancilla] + [x for x in ["a", "b", "c"] if x != target_ancilla]
    op = qt.tensor(U_Sq, *[qt.qeye(2) for _ in labels[2:]])
    order = [labels.index(x) for x in ["S", "a", "b", "c"]]
    return op.permute(order)


def controlled_system_qubit_unitary(
    U_Sq: qt.Qobj,
    target_ancilla: str,
    control_bits: dict[str, int],
    system_dim: int,
) -> qt.Qobj:
    """Conditionally apply a system-plus-ancilla unitary in the three-ancilla register."""
    active = embed_system_qubit_unitary(U_Sq, target_ancilla, system_dim)
    if not control_bits:
        return active
    ident = full_identity_sabc(system_dim)
    P_control = projector_on_ancilla_bits(system_dim, control_bits)
    return P_control * active + (ident - P_control)


def controlled_system_feedback(
    branch_unitaries: list[tuple[dict[str, int], qt.Qobj]],
    system_dim: int,
) -> qt.Qobj:
    """Conditionally apply system feedback unitaries on selected ancilla branches."""
    ident = full_identity_sabc(system_dim)
    out = 0 * ident
    P_total = 0 * ident

    for branch_bits, U_system in branch_unitaries:
        P_branch = projector_on_ancilla_bits(system_dim, branch_bits)
        out += P_branch * qt.tensor(U_system, qt.qeye(2), qt.qeye(2), qt.qeye(2))
        P_total += P_branch

    return out + (ident - P_total)


def branch_system_state(rho_joint: qt.Qobj, system_dim: int, branch_bits: dict[str, int]) -> qt.Qobj:
    """Trace out ancillas after projecting onto one ancilla branch."""
    P_branch = projector_on_ancilla_bits(system_dim, branch_bits)
    return (P_branch * rho_joint * P_branch).ptrace(0)


def run_three_ancilla_recovery_with_ops(
    rho_after_noise: qt.Qobj,
    stage_ops: list[tuple[str, qt.Qobj]],
    *,
    return_records: bool = False,
) -> tuple[qt.Qobj, qt.Qobj] | tuple[qt.Qobj, qt.Qobj, list[dict[str, float]]]:
    """Run the staged three-ancilla coherent recovery and trace out ancillas."""
    zero = qt.basis(2, 0)
    ancilla_000 = qt.tensor(zero, zero, zero)
    rho_joint = qt.tensor(rho_after_noise, ancilla_000 * ancilla_000.dag())
    records = []

    for label, U in stage_ops:
        rho_joint = U * rho_joint * U.dag()
        records.append({"step": label, "trace": float(np.real(rho_joint.tr()))})

    rho_out = rho_joint.ptrace(0)
    if return_records:
        return rho_joint, rho_out, records
    return rho_joint, rho_out


def recovery_stage_ops(
    split_unitaries: list[qt.Qobj],
    feedback_unitaries: list[qt.Qobj],
    *,
    system_dim: Optional[int] = None,
    prefix: str = "exact",
) -> list[tuple[str, qt.Qobj]]:
    """Embed split and feedback unitaries into the three-ancilla recovery circuit."""
    if len(split_unitaries) != 3 or len(feedback_unitaries) != 4:
        raise ValueError("Expected three split unitaries and four feedback unitaries.")
    if system_dim is None:
        system_dim = feedback_unitaries[0].shape[0]

    U_stage0 = controlled_system_qubit_unitary(split_unitaries[0], "a", {}, system_dim)
    F_stage0 = controlled_system_feedback([({"a": 0}, feedback_unitaries[0])], system_dim)
    U_stage1 = controlled_system_qubit_unitary(split_unitaries[1], "b", {"a": 1}, system_dim)
    F_stage1 = controlled_system_feedback([({"a": 1, "b": 0}, feedback_unitaries[1])], system_dim)
    U_stage2 = controlled_system_qubit_unitary(split_unitaries[2], "c", {"a": 1, "b": 1}, system_dim)
    F_stage2 = controlled_system_feedback(
        [
            ({"a": 1, "b": 1, "c": 0}, feedback_unitaries[2]),
            ({"a": 1, "b": 1, "c": 1}, feedback_unitaries[3]),
        ],
        system_dim,
    )
    return [
        (f"{prefix} split 0", U_stage0),
        (f"{prefix} feedback 0", F_stage0),
        (f"{prefix} split 1", U_stage1),
        (f"{prefix} feedback 1", F_stage1),
        (f"{prefix} split 2", U_stage2),
        (f"{prefix} feedback 2/3", F_stage2),
    ]


def build_lv_recovery_data(
    recovery_ops: list[qt.Qobj],
    *,
    reference_weight: Optional[int] = None,
    rcond: float = 1e-12,
    phase_tol: float = 1e-9,
) -> dict[str, Any]:
    """Build exact split/feedback data for four recovery Kraus operators."""
    if len(recovery_ops) != 4:
        raise ValueError("The three-ancilla branching construction currently expects four recovery ops.")

    system_dim = recovery_ops[0].shape[0]
    num_qubits = system_dim - 1
    if reference_weight is None:
        reference_weight = num_qubits

    feedback_ops = []
    B_ops = []
    for R in recovery_ops:
        U_r, B_r = polar_unitary_completion(R, rcond=rcond)
        feedback_ops.append(U_r)
        B_ops.append(B_r)

    B0, B1, B2, B3 = B_ops
    ident = qt.qeye(system_dim)

    S0_full = sqrt_pos(ident - B0 * B0, rcond=rcond)
    U_split0_exact = eq21_unitary_from_B0_S0(B0, S0_full)

    S0_residual = sqrt_pos(B1 * B1 + B2 * B2 + B3 * B3, rcond=rcond)
    S0_inv = pinv_pos(S0_residual, rcond=rcond)
    P0_residual = support_projector(S0_residual, rcond=rcond)
    Q0_residual = ident - P0_residual

    B1_tilde = (B1 * S0_inv) * P0_residual + Q0_residual
    S1_residual = sqrt_pos(B2 * B2 + B3 * B3, rcond=rcond)
    S1_tilde = (S1_residual * S0_inv) * P0_residual
    U_split1_exact = complete_split_unitary(B1_tilde, S1_tilde, rcond=rcond)

    S1_inv = pinv_pos(S1_residual, rcond=rcond)
    P1_residual = support_projector(S1_residual, rcond=rcond)
    Q1_residual = ident - P1_residual

    B2_tilde = (B2 * S1_inv) * P1_residual + Q1_residual
    B3_tilde = (B3 * S1_inv) * P1_residual
    U_split2_exact = complete_split_unitary(B2_tilde, B3_tilde, rcond=rcond)

    E_split0 = B0 + 1j * S0_full
    E_split1 = closest_unitary(B1_tilde + 1j * S1_tilde)
    E_split2 = closest_unitary(B2_tilde + 1j * B3_tilde)
    theta_bg0 = background_phase_from_diagonalizer(E_split0, B0)

    spec_groups = {
        "stage 0 split entangler": spectral_phase_specs_stable(
            E_split0,
            "stage 0 split entangler",
            diagonalizer=B0,
            background_theta=theta_bg0,
            phase_tol=phase_tol,
        ),
        "stage 1 split entangler": spectral_phase_specs_stable(
            E_split1,
            "stage 1 split entangler",
            phase_tol=phase_tol,
        ),
        "stage 2 split entangler": spectral_phase_specs_stable(
            E_split2,
            "stage 2 split entangler",
            phase_tol=phase_tol,
        ),
    }
    for r, U_r in enumerate(feedback_ops):
        spec_groups[f"feedback U_{r}"] = spectral_phase_specs_stable(
            U_r,
            f"feedback U_{r}",
            phase_tol=phase_tol,
        )

    return {
        "R_ops": recovery_ops,
        "feedback_ops": feedback_ops,
        "B_ops": B_ops,
        "U_split_exact": [U_split0_exact, U_split1_exact, U_split2_exact],
        "E_split": [E_split0, E_split1, E_split2],
        "theta_bg0": theta_bg0,
        "spec_groups": spec_groups,
        "reference_weight": reference_weight,
        "num_qubits": num_qubits,
        "system_dim": system_dim,
    }


def default_gpg_recovery_settings(label: str, eig_index: int, counter: int) -> dict[str, int]:
    """Default optimization budget for a GPG recovery sweep target."""
    seed = 5000 + 41 * counter + 13 * int(eig_index)
    if label == "stage 0 split entangler":
        return {"pulses": 9, "restarts": 5, "maxiter": 650, "seed": seed}
    if label == "stage 2 split entangler":
        return {"pulses": 9, "restarts": 4, "maxiter": 520, "seed": seed}
    if label == "feedback U_2":
        return {"pulses": 9, "restarts": 4, "maxiter": 520, "seed": seed}
    if label == "feedback U_0":
        return {"pulses": 9, "restarts": 2, "maxiter": 320, "seed": seed}
    if label in ("stage 1 split entangler", "feedback U_1"):
        return {"pulses": 8, "restarts": 2, "maxiter": 260, "seed": seed}
    return {"pulses": 7, "restarts": 2, "maxiter": 220, "seed": seed}


def optimize_recovery_target(
    num_qubits: int,
    target: ArrayLike,
    settings: dict[str, int],
    ref: qt.Qobj,
    label: str,
    eig_index: int,
    *,
    initial_params: Optional[ArrayLike] = None,
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
) -> dict[str, Any]:
    """Optimize one GPG state-preparation target used by the recovery synthesis."""
    import time

    mode = _validate_gpg_mode_settings(gpg_mode, cooperativity)
    if phase_insensitive_overlap(target, ref) > 1 - 1e-12:
        record = {
            "X": np.empty((0, 4)),
            "F_state": 1.0,
            "1 - F_state": 0.0,
            "Tr_state": 1.0,
            "phase-aligned state error": 0.0,
            "gpg_mode": mode,
            "cooperativity": _recorded_gpg_cooperativity(mode, cooperativity),
            "elapsed_s": 0.0,
            "pulses": 0,
            "restarts": 0,
            "maxiter": 0,
            "source": "reference state",
        }
        if mode == "detuned":
            record["X_detuned"] = np.empty((0, 5))
            record["detunings"] = np.empty((0,), dtype=float)
        return record

    if initial_params is not None:
        settings = dict(settings)
        if mode == "detuned":
            settings["pulses"] = max(settings["pulses"], len(coerce_detuned_pulse_params(initial_params)))
        else:
            settings["pulses"] = max(settings["pulses"], len(coerce_pulse_params(initial_params)))

    t0 = time.perf_counter()
    result = optimize_state_gpg(
        num_qubits,
        target,
        pulses=settings["pulses"],
        initial_state=ref,
        initial_params=initial_params,
        restarts=settings.get("restarts", 2),
        maxiter=settings.get("maxiter", 220),
        seed=settings.get("seed"),
        continuation=initial_params is None,
        gpg_mode=mode,
        cooperativity=cooperativity,
    )
    elapsed = time.perf_counter() - t0
    if mode == "noiseless":
        achieved = apply_gpg_sequence(num_qubits, result.X, ref)
        phase_error = phase_aligned_state_error(target, achieved)
        trace = 1.0
    else:
        phase_error = np.nan
        trace = float(np.real(result.rho_best.tr())) if result.rho_best is not None else np.nan
    record = {
        "X": result.X,
        "F_state": result.Fbest,
        "1 - F_state": result.fbest,
        "Tr_state": trace,
        "phase-aligned state error": phase_error,
        "gpg_mode": mode,
        "cooperativity": _recorded_gpg_cooperativity(mode, cooperativity),
        "elapsed_s": elapsed,
        "pulses": settings["pulses"],
        "restarts": settings.get("restarts", 2),
        "maxiter": settings.get("maxiter", 220),
        "source": f"optimized {label}:{eig_index}" + (" warm-start" if initial_params is not None else ""),
    }
    if mode == "detuned":
        record["X_detuned"] = result.X_with_detuning
        record["detunings"] = result.detunings
    return record


def synthesize_gpg_recovery_from_lv_data(
    data: dict[str, Any],
    *,
    prior_sequences: Optional[dict[tuple[str, int], ArrayLike]] = None,
    settings_fn: Callable[[str, int, int], dict[str, int]] = default_gpg_recovery_settings,
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Synthesize GPG split and feedback unitaries for one LV recovery data set."""
    mode = _validate_gpg_mode_settings(gpg_mode, cooperativity)
    prior_sequences = prior_sequences or {}
    num_qubits = data["num_qubits"]
    reference_weight = data["reference_weight"]
    system_dim = data["system_dim"]
    ref = qt.basis(num_qubits + 1, reference_weight)
    sequence_cache: list[dict[str, Any]] = []
    optimization_records = []
    pulse_sequences = {}
    counter = 0

    for label, specs in data["spec_groups"].items():
        for spec in specs:
            key = (label, int(spec["eig_index"]))
            cached = reusable_sequence_for_target(spec["target"], sequence_cache)
            if cached is not None:
                X = cached.get("X_detuned", cached["X"]) if mode == "detuned" else cached["X"]
                if log is not None:
                    log(f"  reused {label} eig={spec['eig_index']} from {cached['source']}")
                spec.update(
                    state_prep_sequence_record(
                        num_qubits,
                        spec["target"],
                        X,
                        initial_state=ref,
                        gpg_mode=mode,
                        cooperativity=cooperativity,
                        source=f"reused {cached['source']}",
                    )
                )
            else:
                settings = settings_fn(label, int(spec["eig_index"]), counter)
                warm = prior_sequences.get(key)
                warm_msg = " with warm start" if warm is not None else ""
                if log is not None:
                    log(
                        f"  optimizing {label} eig={spec['eig_index']} "
                        f"pulses={settings['pulses']} restarts={settings['restarts']}{warm_msg}"
                    )
                result = optimize_recovery_target(
                    num_qubits,
                    spec["target"],
                    settings,
                    ref,
                    label,
                    int(spec["eig_index"]),
                    initial_params=warm,
                    gpg_mode=mode,
                    cooperativity=cooperativity,
                )
                spec.update(result)
                if log is not None:
                    log(
                        f"    done 1-F_state={spec['1 - F_state']:.3e} "
                        f"elapsed={spec['elapsed_s']:.1f}s"
                    )
                counter += 1

            sequence_cache.append(
                {
                    "target": spec["target"],
                    "X": spec["X"],
                    "X_detuned": spec.get("X_detuned"),
                    "source": spec["source"],
                }
            )
            pulse_sequences[key] = spec.get("X_detuned", spec["X"]) if mode == "detuned" else spec["X"]
            optimization_records.append(
                {k: v for k, v in spec.items() if k not in ("target", "X", "X_detuned", "detunings")}
            )

    C_split0_gpg = controlled_entangler_from_gpg_specs(
        num_qubits,
        data["spec_groups"]["stage 0 split entangler"],
        data["theta_bg0"],
        reference_weight=reference_weight,
    )
    U_split0_gpg = eq41_from_controlled_entangler(C_split0_gpg)
    C_split1_gpg = controlled_entangler_from_gpg_specs(
        num_qubits,
        data["spec_groups"]["stage 1 split entangler"],
        0.0,
        reference_weight=reference_weight,
    )
    U_split1_gpg = eq41_from_controlled_entangler(C_split1_gpg)
    C_split2_gpg = controlled_entangler_from_gpg_specs(
        num_qubits,
        data["spec_groups"]["stage 2 split entangler"],
        0.0,
        reference_weight=reference_weight,
    )
    U_split2_gpg = eq41_from_controlled_entangler(C_split2_gpg)
    feedback_gpg = [
        system_unitary_from_gpg_specs(
            num_qubits,
            data["spec_groups"][f"feedback U_{r}"],
            reference_weight=reference_weight,
        )
        for r in range(4)
    ]

    stage_ops_gpg = recovery_stage_ops(
        [U_split0_gpg, U_split1_gpg, U_split2_gpg],
        feedback_gpg,
        system_dim=system_dim,
        prefix="GPG",
    )
    stage_ops_exact = recovery_stage_ops(
        data["U_split_exact"],
        data["feedback_ops"],
        system_dim=system_dim,
        prefix="exact",
    )

    U_split_targets = [
        data["U_split_exact"][0],
        eq41_from_controlled_entangler(exact_controlled_entangler_from_system_unitary(data["E_split"][1])),
        eq41_from_controlled_entangler(exact_controlled_entangler_from_system_unitary(data["E_split"][2])),
    ]
    split_gpg = [U_split0_gpg, U_split1_gpg, U_split2_gpg]
    synthesis_quality = [
        {
            "factor": f"split {idx}",
            "||GPG - target||": (U_gpg - U_target).norm(),
            "unitarity error": unitary_error(U_gpg),
        }
        for idx, (U_gpg, U_target) in enumerate(zip(split_gpg, U_split_targets))
    ]
    for r, (U_gpg, U_target) in enumerate(zip(feedback_gpg, data["feedback_ops"])):
        synthesis_quality.append(
            {
                "factor": f"feedback U_{r}",
                "||GPG - target||": (U_gpg - U_target).norm(),
                "unitarity error": unitary_error(U_gpg),
            }
        )

    return {
        "stage_ops_exact": stage_ops_exact,
        "stage_ops_gpg": stage_ops_gpg,
        "split_gpg": split_gpg,
        "feedback_gpg": feedback_gpg,
        "optimization_records": optimization_records,
        "synthesis_quality": synthesis_quality,
        "pulse_sequences": pulse_sequences,
        "gpg_mode": mode,
        "cooperativity": _recorded_gpg_cooperativity(mode, cooperativity),
    }


def assemble_gpg_recovery_from_lv_data(data: dict[str, Any]) -> dict[str, Any]:
    """Assemble exact/GPG stage operators after every spec in ``data`` has an ``X`` pulse table."""
    num_qubits = data["num_qubits"]
    reference_weight = data["reference_weight"]
    system_dim = data["system_dim"]

    for label, specs in data["spec_groups"].items():
        for spec in specs:
            if "X" not in spec:
                raise KeyError(f"Missing pulse sequence X for {label} eig={spec['eig_index']}")

    C_split0_gpg = controlled_entangler_from_gpg_specs(
        num_qubits,
        data["spec_groups"]["stage 0 split entangler"],
        data["theta_bg0"],
        reference_weight=reference_weight,
    )
    U_split0_gpg = eq41_from_controlled_entangler(C_split0_gpg)
    C_split1_gpg = controlled_entangler_from_gpg_specs(
        num_qubits,
        data["spec_groups"]["stage 1 split entangler"],
        0.0,
        reference_weight=reference_weight,
    )
    U_split1_gpg = eq41_from_controlled_entangler(C_split1_gpg)
    C_split2_gpg = controlled_entangler_from_gpg_specs(
        num_qubits,
        data["spec_groups"]["stage 2 split entangler"],
        0.0,
        reference_weight=reference_weight,
    )
    U_split2_gpg = eq41_from_controlled_entangler(C_split2_gpg)
    feedback_gpg = [
        system_unitary_from_gpg_specs(
            num_qubits,
            data["spec_groups"][f"feedback U_{r}"],
            reference_weight=reference_weight,
        )
        for r in range(4)
    ]

    stage_ops_gpg = recovery_stage_ops(
        [U_split0_gpg, U_split1_gpg, U_split2_gpg],
        feedback_gpg,
        system_dim=system_dim,
        prefix="GPG",
    )
    stage_ops_exact = recovery_stage_ops(
        data["U_split_exact"],
        data["feedback_ops"],
        system_dim=system_dim,
        prefix="exact",
    )

    U_split_targets = [
        data["U_split_exact"][0],
        eq41_from_controlled_entangler(exact_controlled_entangler_from_system_unitary(data["E_split"][1])),
        eq41_from_controlled_entangler(exact_controlled_entangler_from_system_unitary(data["E_split"][2])),
    ]
    split_gpg = [U_split0_gpg, U_split1_gpg, U_split2_gpg]
    synthesis_quality = [
        {
            "factor": f"split {idx}",
            "||GPG - target||": (U_gpg - U_target).norm(),
            "unitarity error": unitary_error(U_gpg),
        }
        for idx, (U_gpg, U_target) in enumerate(zip(split_gpg, U_split_targets))
    ]
    for r, (U_gpg, U_target) in enumerate(zip(feedback_gpg, data["feedback_ops"])):
        synthesis_quality.append(
            {
                "factor": f"feedback U_{r}",
                "||GPG - target||": (U_gpg - U_target).norm(),
                "unitarity error": unitary_error(U_gpg),
            }
        )

    pulse_sequences = {
        (label, int(spec["eig_index"])): spec["X"]
        for label, specs in data["spec_groups"].items()
        for spec in specs
    }
    return {
        "stage_ops_exact": stage_ops_exact,
        "stage_ops_gpg": stage_ops_gpg,
        "split_gpg": split_gpg,
        "feedback_gpg": feedback_gpg,
        "synthesis_quality": synthesis_quality,
        "pulse_sequences": pulse_sequences,
    }


def recovery_point_from_lv_data(
    data: dict[str, Any],
    rho_after_noise: qt.Qobj,
    reference_rho: qt.Qobj,
    *,
    p: float,
    optimization_records: list[dict[str, Any]],
    logical_kets: Optional[tuple[qt.Qobj, qt.Qobj]] = None,
    fidelity_fn: Optional[Callable[[qt.Qobj], float]] = None,
) -> dict[str, Any]:
    """Evaluate exact and GPG staged recovery after pulse specs are attached to ``data``."""
    assembled = assemble_gpg_recovery_from_lv_data(data)
    _, rho_exact = run_three_ancilla_recovery_with_ops(rho_after_noise, assembled["stage_ops_exact"])
    _, rho_gpg = run_three_ancilla_recovery_with_ops(rho_after_noise, assembled["stage_ops_gpg"])

    exact_fidelity = output_fidelity(
        rho_exact,
        reference_rho,
        logical_kets=logical_kets,
        fidelity_fn=fidelity_fn,
    )
    gpg_fidelity = output_fidelity(
        rho_gpg,
        reference_rho,
        logical_kets=logical_kets,
        fidelity_fn=fidelity_fn,
    )
    output_distance = 0.5 * (rho_gpg - rho_exact).norm()
    synthesis_quality = [{"p": float(p), **record} for record in assembled["synthesis_quality"]]
    metrics = {
        "p": float(p),
        "exact infidelity": float(1 - exact_fidelity),
        "GPG infidelity": float(1 - gpg_fidelity),
        "GPG - exact infidelity penalty": float((1 - gpg_fidelity) - (1 - exact_fidelity)),
        "0.5||rho_GPG-rho_exact||_1": float(output_distance),
        "Tr exact output": float(np.real(rho_exact.tr())),
        "Tr GPG output": float(np.real(rho_gpg.tr())),
        "max state-prep infidelity": float(
            max((r["1 - F_state"] for r in optimization_records), default=0.0)
        ),
        "total optimized seconds": float(sum(r["elapsed_s"] for r in optimization_records)),
    }
    return {
        "metrics": metrics,
        "rho_exact_recovered": rho_exact,
        "rho_gpg_recovered": rho_gpg,
        "optimization_records": optimization_records,
        "synthesis_quality": synthesis_quality,
        "pulse_sequences": assembled["pulse_sequences"],
    }


def refine_noiseless_gpg_recovery_point(
    cache: dict[str, Any],
    p: float,
    rho: qt.Qobj,
    error_model,
    recovery_ops,
    *,
    state_tol: float,
    synth_tol: float,
    mode: str,
    max_targets: Optional[int],
    logical_kets: Optional[tuple[qt.Qobj, qt.Qobj]] = None,
    fidelity_fn: Optional[Callable[[qt.Qobj], float]] = None,
    reference_weight: Optional[int] = None,
    project_top_block: bool = True,
    cache_path: Optional[str | Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    """Re-optimize suspect pulse targets for one cached noiseless GPG recovery point."""
    if rho.isket:
        rho = qt.ket2dm(rho)
    dim = rho.shape[0]
    old_point = cache_point_for_p(cache, p, cache_path=cache_path)
    old_sequences = old_point["pulse_sequences"]

    channel = error_model(p) if callable(error_model) else error_model
    rho_after_noise = apply_channel_to_state(channel, rho, project_top_block=project_top_block)
    R_ops = recovery_ops(p) if callable(recovery_ops) else recovery_ops
    if project_top_block:
        R_ops = restrict_operators_to_dimension(list(R_ops), dim)
    data = build_lv_recovery_data(R_ops, reference_weight=reference_weight)

    suspect_keys = suspect_refinement_keys(
        old_point,
        state_tol=state_tol,
        synth_tol=synth_tol,
    )
    if max_targets is not None and len(suspect_keys) > max_targets:
        old_records = {
            (r["factor"], int(r["eig_index"])): float(r.get("1 - F_state", 0.0))
            for r in old_point.get("optimization_records", [])
        }
        suspect_keys = set(
            sorted(suspect_keys, key=lambda key: old_records.get(key, 0.0), reverse=True)[
                :max_targets
            ]
        )

    ref = qt.basis(data["num_qubits"] + 1, data["reference_weight"])
    optimization_records = []
    sequence_cache: list[dict[str, Any]] = []

    if log is not None:
        log(f"p={p:.12g}: refining {len(suspect_keys)} targets")
    for label, specs in data["spec_groups"].items():
        for spec in specs:
            key = (label, int(spec["eig_index"]))
            old_params = old_sequences.get(key)
            cached = reusable_sequence_for_target(spec["target"], sequence_cache)

            if cached is not None and key not in suspect_keys:
                spec.update(
                    state_prep_sequence_record(
                        data["num_qubits"],
                        spec["target"],
                        cached["X"],
                        initial_state=ref,
                        source=f"reused refined {cached['source']}",
                    )
                )
            elif key in suspect_keys:
                settings = refinement_settings(label, int(spec["eig_index"]), old_params, mode=mode)
                if log is not None:
                    log(
                        f"  optimizing {label} eig={spec['eig_index']} "
                        f"pulses={settings['pulses']} restarts={settings['restarts']} "
                        f"maxiter={settings['maxiter']}"
                    )
                result = optimize_recovery_target(
                    data["num_qubits"],
                    spec["target"],
                    settings,
                    ref,
                    label,
                    int(spec["eig_index"]),
                    initial_params=old_params,
                )
                spec.update(result)

                old_error = None
                for old_record in old_point.get("optimization_records", []):
                    if (
                        old_record["factor"] == label
                        and int(old_record["eig_index"]) == int(spec["eig_index"])
                    ):
                        old_error = float(old_record["1 - F_state"])
                        break
                if log is not None:
                    log(
                        f"    old 1-F={old_error if old_error is not None else float('nan'):.3e}; "
                        f"new 1-F={spec['1 - F_state']:.3e}; "
                        f"elapsed={spec['elapsed_s']:.1f}s"
                    )
            else:
                if old_params is None:
                    raise KeyError(f"No cached sequence for {label} eig={spec['eig_index']}")
                spec.update(
                    state_prep_sequence_record(
                        data["num_qubits"],
                        spec["target"],
                        old_params,
                        initial_state=ref,
                        source="cached",
                    )
                )

            sequence_cache.append({"target": spec["target"], "X": spec["X"], "source": spec["source"]})
            optimization_records.append({k: v for k, v in spec.items() if k not in ("target", "X")})

    new_point = recovery_point_from_lv_data(
        data,
        rho_after_noise,
        rho,
        p=p,
        optimization_records=optimization_records,
        logical_kets=logical_kets,
        fidelity_fn=fidelity_fn,
    )
    return new_point, suspect_keys


def run_gpg_recovery(
    rho: qt.Qobj,
    error_model,
    recovery_ops,
    p: float,
    *,
    logical_kets: Optional[tuple[qt.Qobj, qt.Qobj]] = None,
    fidelity_fn: Optional[Callable[[qt.Qobj], float]] = None,
    reference_weight: Optional[int] = None,
    prior_sequences: Optional[dict[tuple[str, int], ArrayLike]] = None,
    project_top_block: bool = True,
    settings_fn: Callable[[str, int, int], dict[str, int]] = default_gpg_recovery_settings,
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Run exact staged recovery and GPG-synthesized recovery for one ``p`` value."""
    mode = _validate_gpg_mode_settings(gpg_mode, cooperativity)
    if rho.isket:
        rho = qt.ket2dm(rho)
    dim = rho.shape[0]

    channel = error_model(p) if callable(error_model) else error_model
    rho_after_noise = apply_channel_to_state(channel, rho, project_top_block=project_top_block)

    R_ops = recovery_ops(p) if callable(recovery_ops) else recovery_ops
    if project_top_block:
        R_ops = restrict_operators_to_dimension(list(R_ops), dim)
    data = build_lv_recovery_data(R_ops, reference_weight=reference_weight)
    synth = synthesize_gpg_recovery_from_lv_data(
        data,
        prior_sequences=prior_sequences,
        settings_fn=settings_fn,
        gpg_mode=mode,
        cooperativity=cooperativity,
        log=log,
    )

    _, rho_exact = run_three_ancilla_recovery_with_ops(rho_after_noise, synth["stage_ops_exact"])
    _, rho_gpg = run_three_ancilla_recovery_with_ops(rho_after_noise, synth["stage_ops_gpg"])

    exact_fidelity = output_fidelity(
        rho_exact,
        rho,
        logical_kets=logical_kets,
        fidelity_fn=fidelity_fn,
    )
    gpg_fidelity = output_fidelity(
        rho_gpg,
        rho,
        logical_kets=logical_kets,
        fidelity_fn=fidelity_fn,
    )
    output_distance = 0.5 * (rho_gpg - rho_exact).norm()

    synthesis_quality = [
        {"p": float(p), **record}
        for record in synth["synthesis_quality"]
    ]
    metrics = {
        "p": float(p),
        "gpg_mode": mode,
        "GPG cooperativity": _recorded_gpg_cooperativity(mode, cooperativity),
        "exact infidelity": float(1 - exact_fidelity),
        "GPG infidelity": float(1 - gpg_fidelity),
        "GPG - exact infidelity penalty": float(gpg_fidelity - exact_fidelity) * -1,
        "0.5||rho_GPG-rho_exact||_1": float(output_distance),
        "Tr noisy rho": float(np.real(rho_after_noise.tr())),
        "Tr exact output": float(np.real(rho_exact.tr())),
        "Tr GPG output": float(np.real(rho_gpg.tr())),
        "max state-prep infidelity": float(
            max((r["1 - F_state"] for r in synth["optimization_records"]), default=0.0)
        ),
        "total optimized seconds": float(sum(r["elapsed_s"] for r in synth["optimization_records"])),
    }

    return {
        "metrics": metrics,
        "rho_after_noise": rho_after_noise,
        "rho_exact_recovered": rho_exact,
        "rho_gpg_recovered": rho_gpg,
        "lv_data": data,
        "optimization_records": synth["optimization_records"],
        "synthesis_quality": synthesis_quality,
        "pulse_sequences": synth["pulse_sequences"],
        "gpg_mode": mode,
        "cooperativity": _recorded_gpg_cooperativity(mode, cooperativity),
    }


def run_gpg_recovery_sweep(
    rho: qt.Qobj,
    error_model,
    recovery_ops,
    p_values,
    *,
    logical_kets: Optional[tuple[qt.Qobj, qt.Qobj]] = None,
    fidelity_fn: Optional[Callable[[qt.Qobj], float]] = None,
    reference_weight: Optional[int] = None,
    cache_path: Optional[str | Path] = None,
    project_top_block: bool = True,
    settings_fn: Callable[[str, int, int], dict[str, int]] = default_gpg_recovery_settings,
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
    use_cache: bool = True,
    save_cache_each_point: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Run or load a p-sweep for the selected GPG recovery implementation."""
    import pandas as pd

    mode = _validate_gpg_mode_settings(gpg_mode, cooperativity)
    cache = {"points": {}}
    if cache_path is not None and use_cache and Path(cache_path).exists():
        cache = load_gpg_sweep_cache(cache_path)

    prior_sequences = None
    rows = []
    for p in p_values:
        key = gpg_recovery_cache_key(p, gpg_mode=mode, cooperativity=cooperativity)
        if use_cache and key in cache["points"]:
            if log is not None:
                log(f"Using cached p={float(p):.3e} ({mode})")
            point = cache["points"][key]
        else:
            if log is not None:
                log(f"Running p={float(p):.3e} ({mode})")
            point = run_gpg_recovery(
                rho,
                error_model,
                recovery_ops,
                float(p),
                logical_kets=logical_kets,
                fidelity_fn=fidelity_fn,
                reference_weight=reference_weight,
                prior_sequences=prior_sequences,
                project_top_block=project_top_block,
                settings_fn=settings_fn,
                gpg_mode=mode,
                cooperativity=cooperativity,
                log=log,
            )
            cache["points"][key] = point
            if cache_path is not None and save_cache_each_point:
                save_gpg_sweep_cache(cache, cache_path)

        rows.append(point["metrics"])
        prior_sequences = point["pulse_sequences"]

    if cache_path is not None and not save_cache_each_point:
        save_gpg_sweep_cache(cache, cache_path)

    results = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    return {"results": results, "cache": cache}


def run_noiseless_gpg_recovery(*args, **kwargs) -> dict[str, Any]:
    """
    Backward-compatible alias for :func:`run_gpg_recovery`.

    Prefer ``run_gpg_recovery`` for new notebooks because the implementation now
    supports both noiseless and erroneous finite-cooperativity GPG optimization.
    """
    return run_gpg_recovery(*args, **kwargs)


def run_noiseless_gpg_recovery_sweep(*args, **kwargs) -> dict[str, Any]:
    """
    Backward-compatible alias for :func:`run_gpg_recovery_sweep`.

    Prefer ``run_gpg_recovery_sweep`` for new notebooks.
    """
    return run_gpg_recovery_sweep(*args, **kwargs)


def _attach_existing_sequence_metrics(
    num_qubits: int,
    reference_weight: int,
    spec: dict[str, Any],
    old_params: ArrayLike,
    source: str,
) -> None:
    if old_params is None:
        raise KeyError(f"No cached sequence for {spec['factor']} eig={spec['eig_index']}")
    ref = qt.basis(num_qubits + 1, reference_weight)
    spec.update(
        state_prep_sequence_record(
            num_qubits,
            spec["target"],
            old_params,
            initial_state=ref,
            source=source,
        )
    )


def build_gpg_recovery_point_from_specs(
    ns: dict[str, Any],
    p: float,
    data: dict[str, Any],
    optimization_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Rebuild one exact-AD recovery sweep point from already-selected GPG specs.

    ``ns`` is the namespace loaded from the construction notebook; it supplies
    the recovery-specific embedding functions while this module owns the GPG
    bookkeeping and fidelity calculation.
    """
    N = ns["N"]
    system_dim = ns["system_dim"]
    rho_bgm = ns["rho_bgm"]
    l0_bgm = ns["l0_bgm"]
    l1_bgm = ns["l1_bgm"]

    C_split0_gpg = ns["controlled_entangler_from_sweep_specs"](
        data["spec_groups"]["stage 0 split entangler"],
        data["theta_bg0"],
    )
    U_split0_gpg = ns["eq41_from_controlled_entangler"](C_split0_gpg)

    C_split1_gpg = ns["controlled_entangler_from_sweep_specs"](
        data["spec_groups"]["stage 1 split entangler"],
        0.0,
    )
    U_split1_gpg = ns["eq41_from_controlled_entangler"](C_split1_gpg)

    C_split2_gpg = ns["controlled_entangler_from_sweep_specs"](
        data["spec_groups"]["stage 2 split entangler"],
        0.0,
    )
    U_split2_gpg = ns["eq41_from_controlled_entangler"](C_split2_gpg)

    feedback_gpg = [
        ns["system_unitary_from_sweep_specs"](data["spec_groups"][f"feedback U_{r}"], 0.0)
        for r in range(4)
    ]

    U_stage0_gpg = ns["controlled_system_qubit_unitary"](U_split0_gpg, "a", {}, system_dim)
    F_stage0_gpg = ns["controlled_system_feedback"]([({"a": 0}, feedback_gpg[0])], system_dim)
    U_stage1_gpg = ns["controlled_system_qubit_unitary"](
        U_split1_gpg,
        "b",
        {"a": 1},
        system_dim,
    )
    F_stage1_gpg = ns["controlled_system_feedback"](
        [({"a": 1, "b": 0}, feedback_gpg[1])],
        system_dim,
    )
    U_stage2_gpg = ns["controlled_system_qubit_unitary"](
        U_split2_gpg,
        "c",
        {"a": 1, "b": 1},
        system_dim,
    )
    F_stage2_gpg = ns["controlled_system_feedback"](
        [
            ({"a": 1, "b": 1, "c": 0}, feedback_gpg[2]),
            ({"a": 1, "b": 1, "c": 1}, feedback_gpg[3]),
        ],
        system_dim,
    )

    stage_ops_gpg = [
        ("GPG split 0", U_stage0_gpg),
        ("GPG feedback 0", F_stage0_gpg),
        ("GPG split 1", U_stage1_gpg),
        ("GPG feedback 1", F_stage1_gpg),
        ("GPG split 2", U_stage2_gpg),
        ("GPG feedback 2/3", F_stage2_gpg),
    ]

    U_stage0_exact = ns["controlled_system_qubit_unitary"](
        data["U_split_exact"][0],
        "a",
        {},
        system_dim,
    )
    F_stage0_exact = ns["controlled_system_feedback"](
        [({"a": 0}, data["feedback_ops"][0])],
        system_dim,
    )
    U_stage1_exact = ns["controlled_system_qubit_unitary"](
        data["U_split_exact"][1],
        "b",
        {"a": 1},
        system_dim,
    )
    F_stage1_exact = ns["controlled_system_feedback"](
        [({"a": 1, "b": 0}, data["feedback_ops"][1])],
        system_dim,
    )
    U_stage2_exact = ns["controlled_system_qubit_unitary"](
        data["U_split_exact"][2],
        "c",
        {"a": 1, "b": 1},
        system_dim,
    )
    F_stage2_exact = ns["controlled_system_feedback"](
        [
            ({"a": 1, "b": 1, "c": 0}, data["feedback_ops"][2]),
            ({"a": 1, "b": 1, "c": 1}, data["feedback_ops"][3]),
        ],
        system_dim,
    )
    stage_ops_exact = [
        ("exact split 0", U_stage0_exact),
        ("exact feedback 0", F_stage0_exact),
        ("exact split 1", U_stage1_exact),
        ("exact feedback 1", F_stage1_exact),
        ("exact split 2", U_stage2_exact),
        ("exact feedback 2/3", F_stage2_exact),
    ]

    rho_after_exact_p = ns["apply_exact_global_ad_dicke"](rho_bgm, N, float(p), 1.0)
    _, rho_exact_recovered = ns["run_sweep_recovery_with_ops"](rho_after_exact_p, stage_ops_exact)
    _, rho_gpg_recovered = ns["run_sweep_recovery_with_ops"](rho_after_exact_p, stage_ops_gpg)

    exact_infidelity = 1 - codespace_mixed_fidelity(rho_exact_recovered, l0_bgm, l1_bgm)
    gpg_infidelity = 1 - codespace_mixed_fidelity(rho_gpg_recovered, l0_bgm, l1_bgm)
    output_distance = 0.5 * (rho_gpg_recovered - rho_exact_recovered).norm()

    U_split_targets = [
        data["U_split_exact"][0],
        ns["eq41_from_controlled_entangler"](
            ns["exact_controlled_entangler_from_system_unitary_sweep"](data["E_split"][1])
        ),
        ns["eq41_from_controlled_entangler"](
            ns["exact_controlled_entangler_from_system_unitary_sweep"](data["E_split"][2])
        ),
    ]
    split_gpg = [U_split0_gpg, U_split1_gpg, U_split2_gpg]

    synthesis_quality = [
        {
            "p": float(p),
            "factor": f"split {idx}",
            "||GPG - target||": (U_gpg - U_target).norm(),
            "unitarity error": ns["unitary_error"](U_gpg),
        }
        for idx, (U_gpg, U_target) in enumerate(zip(split_gpg, U_split_targets))
    ]
    for r, (U_gpg, U_target) in enumerate(zip(feedback_gpg, data["feedback_ops"])):
        synthesis_quality.append(
            {
                "p": float(p),
                "factor": f"feedback U_{r}",
                "||GPG - target||": (U_gpg - U_target).norm(),
                "unitarity error": ns["unitary_error"](U_gpg),
            }
        )

    pulse_sequences = {
        (label, int(spec["eig_index"])): spec["X"]
        for label, specs in data["spec_groups"].items()
        for spec in specs
    }

    metrics = {
        "p": float(p),
        "exact infidelity": float(exact_infidelity),
        "GPG infidelity": float(gpg_infidelity),
        "GPG - exact infidelity penalty": float(gpg_infidelity - exact_infidelity),
        "0.5||rho_GPG-rho_exact||_1": float(output_distance),
        "Tr exact output": float(np.real(rho_exact_recovered.tr())),
        "Tr GPG output": float(np.real(rho_gpg_recovered.tr())),
        "max state-prep infidelity": float(
            max((r["1 - F_state"] for r in optimization_records), default=0.0)
        ),
        "total optimized seconds": float(sum(r["elapsed_s"] for r in optimization_records)),
    }

    return {
        "metrics": metrics,
        "optimization_records": optimization_records,
        "synthesis_quality": synthesis_quality,
        "pulse_sequences": pulse_sequences,
    }


def refine_gpg_recovery_point(
    ns: dict[str, Any],
    cache: dict[str, Any],
    p: float,
    *,
    state_tol: float,
    synth_tol: float,
    mode: str,
    max_targets: Optional[int],
    cache_path: Optional[str | Path] = None,
    log=None,
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    """Warm-start and refine the suspect GPG targets for one sweep point."""
    old_point = cache_point_for_p(cache, p, cache_path=cache_path)
    data = ns["build_exact_lv_data_for_p"](p)
    old_sequences = old_point["pulse_sequences"]

    suspect_keys = suspect_refinement_keys(
        old_point,
        state_tol=state_tol,
        synth_tol=synth_tol,
    )
    if max_targets is not None and len(suspect_keys) > max_targets:
        old_records = {
            (r["factor"], int(r["eig_index"])): float(r.get("1 - F_state", 0.0))
            for r in old_point.get("optimization_records", [])
        }
        suspect_keys = set(
            sorted(suspect_keys, key=lambda key: old_records.get(key, 0.0), reverse=True)[
                :max_targets
            ]
        )

    ref = qt.basis(ns["N"] + 1, ns["reference_weight"])
    optimization_records = []
    sequence_cache = []

    if log is not None:
        log(f"p={p:.12g}: refining {len(suspect_keys)} targets")
    for label, specs in data["spec_groups"].items():
        for spec in specs:
            key = (label, int(spec["eig_index"]))
            old_params = old_sequences.get(key)

            cached = reusable_sequence_for_target(spec["target"], sequence_cache)
            if cached is not None and key not in suspect_keys:
                _attach_existing_sequence_metrics(
                    ns["N"],
                    ns["reference_weight"],
                    spec,
                    cached["X"],
                    f"reused refined {cached['source']}",
                )
            elif key in suspect_keys:
                settings = refinement_settings(label, int(spec["eig_index"]), old_params, mode=mode)
                if log is not None:
                    log(
                        f"  optimizing {label} eig={spec['eig_index']} "
                        f"pulses={settings['pulses']} restarts={settings['restarts']} "
                        f"maxiter={settings['maxiter']}"
                    )
                result = ns["optimize_sweep_target"](
                    spec["target"],
                    settings,
                    ref,
                    label,
                    int(spec["eig_index"]),
                    initial_params=old_params,
                )
                spec.update(result)

                old_error = None
                for old_record in old_point.get("optimization_records", []):
                    if (
                        old_record["factor"] == label
                        and int(old_record["eig_index"]) == int(spec["eig_index"])
                    ):
                        old_error = float(old_record["1 - F_state"])
                        break
                if log is not None:
                    log(
                        f"    old 1-F={old_error if old_error is not None else float('nan'):.3e}; "
                        f"new 1-F={spec['1 - F_state']:.3e}; "
                        f"elapsed={spec['elapsed_s']:.1f}s"
                    )
            else:
                _attach_existing_sequence_metrics(
                    ns["N"],
                    ns["reference_weight"],
                    spec,
                    old_params,
                    "cached",
                )

            sequence_cache.append({"target": spec["target"], "X": spec["X"], "source": spec["source"]})
            optimization_records.append({k: v for k, v in spec.items() if k not in ("target", "X")})

    new_point = build_gpg_recovery_point_from_specs(ns, p, data, optimization_records)
    return new_point, suspect_keys


def redraw_gpg_recovery_sweep_figure(
    cache: dict[str, Any],
    fig_dir: str | Path,
    *,
    basename: str = "noiseless_gpg_implementation",
) -> Path:
    """Redraw the exact-vs-GPG recovery sweep figure from a cache."""
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    sweep_results = gpg_sweep_rows(cache)
    fig, ax = plt.subplots(figsize=(3.45, 2.55), dpi=220)
    ax.loglog(
        sweep_results["p"],
        sweep_results["exact infidelity"],
        "o-",
        lw=1.4,
        ms=4.2,
        label="exact recovery",
    )
    ax.loglog(
        sweep_results["p"],
        sweep_results["GPG infidelity"],
        "s--",
        lw=1.4,
        ms=4.0,
        label="GPG recovery",
    )
    ax.set_xlabel(r"$p = \gamma \Delta t$")
    ax.set_ylabel(r"$1 - F$")
    ax.grid(True, which="both", ls=":", lw=0.55, alpha=0.65)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()

    pdf_path = fig_dir / f"{basename}.pdf"
    png_path = fig_dir / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def state_prep_objective(
    params: ArrayLike,
    num_qubits: int,
    target_state: ArrayLike,
    *,
    initial_state: Optional[ArrayLike] = None,
    pulses: Optional[int] = None,
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
) -> float:
    """
    State-prep objective for noiseless or erroneous GPG pulse maps.

    Noiseless mode uses the MATLAB-style smooth objective
    ``2 * (1 - |<target|psi>|)``.  Erroneous mode propagates a density matrix
    and minimizes ``1 - <target|rho|target>`` with the trace-decreasing error
    model left unnormalized.
    """
    target = normalize_state(target_state)
    psi0 = dicke_basis_state(num_qubits, 0) if initial_state is None else initial_state
    mode = _validate_gpg_mode_settings(gpg_mode, cooperativity)
    if pulses is not None:
        stride = 5 if mode == "detuned" else 4
        params = np.asarray(params, dtype=float)[: stride * pulses]
    if mode == "erroneous":
        fidelity, _ = state_prep_density_fidelity(
            num_qubits,
            params,
            target,
            initial_state=psi0,
            cooperativity=cooperativity,
        )
        return float(1 - min(1.0, fidelity))
    if mode == "detuned":
        fidelity, _ = detuned_state_prep_density_fidelity(
            num_qubits,
            params,
            target,
            initial_state=psi0,
            cooperativity=cooperativity,
        )
        return float(1 - min(1.0, fidelity))

    psi = apply_gpg_sequence(num_qubits, params, psi0).full().ravel()
    return float(2 * (1 - abs(np.vdot(target, psi))))


def state_infidelity(
    params: ArrayLike,
    num_qubits: int,
    target_state: ArrayLike,
    *,
    initial_state: Optional[ArrayLike] = None,
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
) -> float:
    """Return target-state infidelity for a GPG state-prep sequence."""
    target = normalize_state(target_state)
    psi0 = dicke_basis_state(num_qubits, 0) if initial_state is None else initial_state
    mode = _validate_gpg_mode_settings(gpg_mode, cooperativity)
    if mode == "erroneous":
        fidelity, _ = state_prep_density_fidelity(
            num_qubits,
            params,
            target,
            initial_state=psi0,
            cooperativity=cooperativity,
        )
        return float(1 - min(1.0, fidelity))
    if mode == "detuned":
        fidelity, _ = detuned_state_prep_density_fidelity(
            num_qubits,
            params,
            target,
            initial_state=psi0,
            cooperativity=cooperativity,
        )
        return float(1 - min(1.0, fidelity))

    psi = apply_gpg_sequence(num_qubits, params, psi0).full().ravel()
    return float(1 - abs(np.vdot(target, psi)) ** 2)


def _bounds_for_pulses(
    pulses: int,
    *,
    angle_max: float,
    beta_max: float,
    kappa_max: float,
):
    bounds = []
    for _ in range(pulses):
        bounds.extend(
            [
                (-angle_max, angle_max),
                (-beta_max, beta_max),
                (-angle_max, angle_max),
                (-kappa_max, kappa_max),
            ]
        )
    return bounds


def _detuned_bounds_for_pulses(
    pulses: int,
    *,
    angle_max: float,
    beta_max: float,
    kappa_max: float,
    detuning_min: float,
    detuning_max: float,
):
    bounds = []
    for _ in range(pulses):
        bounds.extend(
            [
                (-angle_max, angle_max),
                (-beta_max, beta_max),
                (-angle_max, angle_max),
                (-kappa_max, kappa_max),
                (detuning_min, detuning_max),
            ]
        )
    return bounds


def optimize_state_gpg(
    num_qubits: int,
    target_state: ArrayLike,
    pulses: int,
    *,
    initial_state: Optional[ArrayLike] = None,
    initial_params: Optional[ArrayLike] = None,
    restarts: int = 50,
    maxiter: int = 1500,
    seed: Optional[int] = None,
    continuation: bool = True,
    angle_max: float = np.pi,
    beta_max: float = np.pi,
    kappa_max: float = np.pi,
    detuning_min: float = 0.05,
    detuning_max: float = 20.0,
    detuning_seed: float = 1.0,
    tol: float = 1e-12,
    method: str = "L-BFGS-B",
    gpg_mode: str = "noiseless",
    cooperativity: Optional[float] = None,
) -> GPGStatePrepResult:
    """
    Optimize a GPG sequence that prepares ``target_state``.

    This is the Python counterpart of ``state_prepe_gpg.m``. It uses SciPy's
    bounded local optimizer with random restarts and optional continuation in
    the pulse count.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("optimize_state_gpg requires scipy.optimize.") from exc

    if pulses < 1:
        raise ValueError("pulses must be at least 1.")
    if restarts < 1:
        raise ValueError("restarts must be at least 1.")

    mode = _validate_gpg_mode_settings(gpg_mode, cooperativity)
    stride = 5 if mode == "detuned" else 4
    rng = np.random.default_rng(seed)
    target = normalize_state(target_state)
    psi0 = dicke_basis_state(num_qubits, 0) if initial_state is None else initial_state

    initial_guess = None
    if initial_params is not None:
        initial_guess = np.zeros(stride * pulses, dtype=float)
        if mode == "detuned":
            provided = coerce_detuned_pulse_params(initial_params, detuning_seed=detuning_seed).reshape(-1)
            initial_guess[4::5] = detuning_seed
        else:
            provided = coerce_pulse_params(initial_params).reshape(-1)
        n_copy = min(initial_guess.size, provided.size)
        initial_guess[:n_copy] = provided[:n_copy]

    xbest_full = np.zeros(stride * pulses, dtype=float)
    if mode == "detuned":
        xbest_full[4::5] = detuning_seed
    if initial_guess is not None:
        xbest_full[:] = initial_guess
    trace: list[dict[str, Any]] = []
    p_start = 1 if continuation else pulses

    for P in range(p_start, pulses + 1):
        dim = stride * P
        if mode == "detuned":
            bounds = _detuned_bounds_for_pulses(
                P,
                angle_max=angle_max,
                beta_max=beta_max,
                kappa_max=kappa_max,
                detuning_min=detuning_min,
                detuning_max=detuning_max,
            )
        else:
            bounds = _bounds_for_pulses(
                P,
                angle_max=angle_max,
                beta_max=beta_max,
                kappa_max=kappa_max,
            )
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])

        x_seed = np.zeros(dim, dtype=float)
        if mode == "detuned":
            x_seed[4::5] = detuning_seed
        if initial_guess is not None:
            x_seed[:] = initial_guess[:dim]
        elif P > p_start:
            x_seed[: stride * (P - 1)] = xbest_full[: stride * (P - 1)]
        elif not continuation:
            x_seed[:] = 0
            if mode == "detuned":
                x_seed[4::5] = detuning_seed

        starts = [x_seed]
        for _ in range(1, restarts):
            if rng.random() < 0.5:
                xr = x_seed + 0.25 * (ub - lb) * rng.standard_normal(dim)
                xr = np.clip(xr, lb, ub)
            else:
                xr = lb + (ub - lb) * rng.random(dim)
            starts.append(xr)

        def objective(x):
            return state_prep_objective(
                x,
                num_qubits,
                target,
                initial_state=psi0,
                pulses=P,
                gpg_mode=mode,
                cooperativity=cooperativity,
            )

        best_fun = np.inf
        best_x = None
        best_result = None
        for x0 in starts:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                tol=tol,
                options={"maxiter": maxiter},
            )
            if result.fun < best_fun:
                best_fun = float(result.fun)
                best_x = np.asarray(result.x, dtype=float)
                best_result = result

        if best_x is None:
            raise RuntimeError("No optimization result was produced.")

        xbest_full[:dim] = best_x
        if dim < len(xbest_full):
            xbest_full[dim:] = 0

        if mode == "noiseless":
            psi_P = apply_gpg_sequence(num_qubits, best_x, psi0).full().ravel()
            F_P = abs(np.vdot(target, psi_P)) ** 2
            tr_P = 1.0
        elif mode == "erroneous":
            F_P, tr_P = state_prep_density_fidelity(
                num_qubits,
                best_x,
                target,
                initial_state=psi0,
                cooperativity=cooperativity,
            )
        else:
            F_P, tr_P = detuned_state_prep_density_fidelity(
                num_qubits,
                best_x,
                target,
                initial_state=psi0,
                cooperativity=cooperativity,
            )
        trace.append(
            {
                "pulses": P,
                "objective": best_fun,
                "fidelity": float(F_P),
                "trace": float(tr_P),
                "gpg_mode": mode,
                "success": bool(best_result.success) if best_result is not None else False,
                "message": str(best_result.message) if best_result is not None else "",
            }
        )

    if mode == "detuned":
        X_with_detuning = xbest_full.reshape(-1, 5)
        X_coherent = X_with_detuning[:, :4].copy()
        detunings = X_with_detuning[:, 4].copy()
    else:
        X_with_detuning = None
        X_coherent = xbest_full.reshape(-1, 4)
        detunings = None

    psi_best = apply_gpg_sequence(num_qubits, X_coherent, psi0).full().ravel()
    rho_best = None
    if mode == "noiseless":
        overlap = np.vdot(target, psi_best)
        Fbest = float(abs(overlap) ** 2)
        objbest = float(2 * (1 - abs(overlap)))
    elif mode == "erroneous":
        rho_best = apply_gpg_sequence_density(
            num_qubits,
            xbest_full,
            psi0,
            cooperativity=cooperativity,
            return_qobj=True,
        )
        Fbest, _ = state_prep_density_fidelity(
            num_qubits,
            xbest_full,
            target,
            initial_state=psi0,
            cooperativity=cooperativity,
        )
        Fbest = float(min(1.0, Fbest))
        objbest = float(1 - Fbest)
    else:
        rho_best = apply_detuned_gpg_sequence_density(
            num_qubits,
            X_with_detuning,
            psi0,
            cooperativity=cooperativity,
            return_qobj=True,
        )
        Fbest, _ = detuned_state_prep_density_fidelity(
            num_qubits,
            X_with_detuning,
            target,
            initial_state=psi0,
            cooperativity=cooperativity,
        )
        Fbest = float(min(1.0, Fbest))
        objbest = float(1 - Fbest)

    return GPGStatePrepResult(
        xbest=xbest_full,
        X=X_coherent,
        Fbest=Fbest,
        fbest=1 - Fbest,
        objbest=objbest,
        psi_best=psi_best,
        psi_target=target,
        m_values=dicke_m_values(num_qubits),
        trace=trace,
        rho_best=rho_best,
        gpg_mode=mode,
        cooperativity=_recorded_gpg_cooperativity(mode, cooperativity),
        X_with_detuning=X_with_detuning,
        detunings=detunings,
    )


def rank_one_phase(state: ArrayLike, phase: float) -> qt.Qobj:
    """Return ``I + (exp(i phase)-1)|state><state|``."""
    ket_vec = normalize_state(state).reshape(-1, 1)
    dim = ket_vec.shape[0]
    ket = qt.Qobj(ket_vec, dims=[[dim], [1]])
    P = ket * ket.dag()
    return qt.qeye(dim) + (np.exp(1j * phase) - 1) * P


def phase_from_prep_unitary(
    prep_unitary: qt.Qobj,
    phase: float,
    *,
    reference_weight: int = 0,
    prep_maps_reference_to_target: bool = True,
) -> qt.Qobj:
    """
    Build a rank-one phase from a state-preparation unitary.

    If ``prep_maps_reference_to_target`` is true, ``prep_unitary |D_ref>`` is
    the target vector and this returns ``W P_ref(phi) W^dag``. If false, it
    returns ``W^dag P_ref(phi) W``.
    """
    dim = prep_unitary.shape[0]
    if reference_weight < 0 or reference_weight >= dim:
        raise ValueError(f"reference_weight must be between 0 and {dim - 1}.")
    ref = qt.basis(dim, reference_weight)
    P_ref = qt.qeye(dim) + (np.exp(1j * phase) - 1) * (ref * ref.dag())
    if prep_maps_reference_to_target:
        return prep_unitary * P_ref * prep_unitary.dag()
    return prep_unitary.dag() * P_ref * prep_unitary


def projected_gate_error(
    implemented: ArrayLike,
    target: ArrayLike,
    projector: Optional[ArrayLike] = None,
) -> float:
    """
    Return ``0.5 * ||target - P implemented P||_F^2``.

    This mirrors the main objective used in ``value_err_H.m``.
    """
    U = implemented.full() if isinstance(implemented, qt.Qobj) else np.asarray(implemented, dtype=complex)
    T = target.full() if isinstance(target, qt.Qobj) else np.asarray(target, dtype=complex)
    if projector is None:
        P = np.eye(T.shape[0], dtype=complex)
    else:
        P = projector.full() if isinstance(projector, qt.Qobj) else np.asarray(projector, dtype=complex)
    diff = T - P @ U @ P
    return float(0.5 * np.trace(diff.conj().T @ diff).real)
