import numpy as np
import qutip 

def gross_13(phi: float = 0.0):
    dim = 14

    coeff_1 = np.sqrt(910) / 56
    coeff_2 = -3 * np.sqrt(154) / 56
    coeff_3 = -np.sqrt(770) / 56
    coeff_4 = np.sqrt(70) / 56

    coeff_5 = np.sqrt(231) / 84
    coeff_6 = np.sqrt(1365) / 84
    coeff_7 = -np.sqrt(273) / 28
    coeff_8 = -np.sqrt(3003) / 84

    basis1 = np.zeros(dim, dtype=complex); basis1[13] = 1.0
    basis2 = np.zeros(dim, dtype=complex); basis2[9]  = 1.0
    basis3 = np.zeros(dim, dtype=complex); basis3[5]  = 1.0
    basis4 = np.zeros(dim, dtype=complex); basis4[1]  = 1.0

    state_1 = coeff_1 * basis1 + coeff_2 * basis2 + coeff_3 * basis3 + coeff_4 * basis4
    state_2 = coeff_5 * basis1 + coeff_6 * basis2 + coeff_7 * basis3 + coeff_8 * basis4

    phase_factor = np.exp(1j * phi)
    coeff_comb_1 = np.sqrt(105) / 14
    coeff_comb_2 = np.sqrt(91) / 14

    ket0_14 = coeff_comb_1 * state_1 + coeff_comb_2 * phase_factor * state_2
    ket1_14 = coeff_comb_2 * state_1 - coeff_comb_1 * np.conj(phase_factor) * state_2

    rho14 = 0.5 * (np.outer(ket0_14, np.conj(ket0_14)) + np.outer(ket1_14, np.conj(ket1_14)))

    # --- embed into PIQS reduced space (56) with blocks [14,12,10,8,6,4,2] ---
    total = 56
    rho56 = np.zeros((total, total), dtype=complex)
    rho56[:14, :14] = rho14

    ket0_56 = np.zeros((total, 1), dtype=complex)
    ket1_56 = np.zeros((total, 1), dtype=complex)
    ket0_56[:14, 0] = ket0_14
    ket1_56[:14, 0] = ket1_14

    rho = qutip.Qobj(rho56, dims=[[56], [56]])
    l0  = qutip.Qobj(ket0_56, dims=[[56], [1]])
    l1  = qutip.Qobj(ket1_56, dims=[[56], [1]])
    
    return rho, l0, l1

