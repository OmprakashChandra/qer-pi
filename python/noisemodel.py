import numpy as np
import qutip 
from qutip.piqs.piqs import Dicke
from typing import List

def noisemodel(noise_type: str,  num_qubits: int, gamma: float, dt: float): 
    """ Function to generate a set of Kraus operators or Choi matrix representing 
    a quantum noise channel.
    
    Parameters
    ----------
    noise_type : str 
        Type of noise channel. Supported types: 'global symmetric depolarizing', 'local symmetric depolarizing'. 
    num_qubits : int 
        Number of qubits describing the system. 
    gamma : float 
        Noise rate parameter.
    dt : float
        Time step for the noise channel.

    Returns
    -------
    Kraus operators or Choi matrix representing the noise channel.        
    """
    
    if noise_type == 'global symmetric depolarizing':  
        kraus = global_symmetric_depolarizing (num_qubits,gamma, dt)
        
    elif noise_type == 'local symmetric depolarizing':
        kraus = local_symmetric_depolarizing (num_qubits, gamma, dt)
    else : 
        raise ValueError(f" Unsupported noise type: {noise_type}. Supported types are 'global symmetric depolarizing , 'local symmetric depolarizing' ")   

    return kraus 

def global_symmetric_depolarizing(num_qubits: int, gamma: float, dt: float): 
    """ Generates global symmetric depolarizing channel in the (N+1)-dimensional Dicke space """
    
    system = Dicke(num_qubits)
    system.collective_emission = gamma # global incoherent emission
    system.collective_pumping = gamma # global incoherent pumping
    system.collective_dephasing = gamma # global dephasing
    L = system.liouvillian() 
    d_out = d_in = int(np.sqrt(L.shape[0]))  # d_out and d_in will be used to generate kraus operators from choi of the channel
    E = (L * dt).expm()    # E = exp(L dt)
    choi = qutip.to_choi (E) # convert superoperator E to Choi matrix
    Ks = _kraus_from_choi (choi, d_out, d_in)

    return Ks


def local_symmetric_depolarizing (num_qubits :int, gamma: float, dt: float ): 
    """ Generates local symmetric depolarizing channel """ 

    system = Dicke(num_qubits)
    system.emission = gamma  # local incoherent emission
    system.pumping = gamma  # local incoherent pumping
    system.dephasing = gamma # local dephasing
    L = system.liouvillian()
    d_out = d_in = int(np.sqrt(L.shape[0])) # d_out and d_in will be used to generate kraus operators from choi of the channel     
    E = (L * dt).expm()   # E = exp(L dt)
    choi = qutip.to_choi(E) # convert superoperator E to Choi matrix

    Ks = _kraus_from_choi (choi, d_out, d_in )
    
    return Ks       

def _kraus_from_choi (choi: qutip.Qobj, d_out: int, d_in: int) -> list[qutip.Qobj]: 
    """ converts a choi matrix to a list of kraus operators upto some tolerance.
     The kraus operators will be of dimension:(d_out x d_in). """
    tol = 1e-8
    choi = (choi + choi.dag())/2 # ensuring hermiticity
    w, v = np.linalg.eigh(choi.full())
    w = np.where( w > tol, w, 0.0)
    Ks = []
    for k in range(len(w)): 
        if w[k] <= 0: 
            continue
        vec = np.sqrt(w[k])*v[:,k]
        Kmat = vec.reshape((d_out, d_in), order = 'F') #column stacking
        Ks.append(qutip.Qobj(Kmat, dims = [[d_out],[d_in]]))
    return Ks          


               


