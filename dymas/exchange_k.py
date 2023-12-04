# Calculates Exchange Kernel for plane waves

import numpy as np
import numpy.typing as npt

def Exchange_Kernel_K(lz: int, dz: float, k: npt.NDArray,
                      A_ex: float) -> npt.NDArray:
    '''
    K_ex = 2 A_ex (d²/dx² + d²/dy² + d²/dz²)
    '''
    

    sqr_k = np.linalg.norm(k)**2

    # Kronecker delta
    kd = np.eye(lz+2, dtype='int8')[:, 1:-1]

    K_exh = (+ (2*kd[1:-1] - kd[0:-2] - kd[2:])/dz**2
             + sqr_k * kd[1:-1] )
    #Fix the edges
    K_exh[0,0] -= 1/dz**2
    K_exh[lz-1,lz-1] -= 1/dz**2

    return np.einsum('ij,ab->iajb',
                     -K_exh,
                     np.eye(3))*2*A_ex
