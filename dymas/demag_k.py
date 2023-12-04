# Demagnetizing tensor calculations, using plane waves

import numpy as np
import numpy.typing as npt


def Demag_Kernel_K(lz: int, dz: float, k: npt.NDArray,
                   dtype: str = 'complex128') -> npt.NDArray:
    '''
    Returns the demagnetization kernel K[i,a,j,b](k)
    for a film discretized in lz parts, with height dz,
    and for a wavevector kda
    '''

    grid = np.mgrid[0:lz]
    ZJ_Zi = dz*(grid[:, None] - grid[None, :])

    abs_k = np.linalg.norm(k)
    kDz = abs_k*dz
    sqr_k = abs_k**2

    # Exponential times cosh term (only for Zj-Zi != 0)
    exp_cosh = np.exp(-abs_k*np.abs(ZJ_Zi)) * (np.cosh(kDz) - 1) / kDz

    # Exponential minus one over kDz term
    exp_kDz = (np.exp(-kDz) - 1)/kDz

    # allocate memory
    kernel = np.zeros((lz, 3, lz, 3), dtype=dtype)
    kernel_view = np.einsum('iajb->abij', kernel)

    # fill up the values
    out_ab_common = exp_cosh / sqr_k
    np.fill_diagonal(out_ab_common, (1+exp_kDz)/sqr_k)
    kernel_view[0, 0] = k[0]*k[0] * out_ab_common
    kernel_view[0, 1] = k[0]*k[1] * out_ab_common
    kernel_view[1, 0] = k[1]*k[0] * out_ab_common
    kernel_view[1, 1] = k[1]*k[1] * out_ab_common

    out_a2_common = 1j * np.sign(ZJ_Zi) * exp_cosh / abs_k
    np.fill_diagonal(out_a2_common, 0)
    kernel_view[0, 2] = k[0] * out_a2_common
    kernel_view[1, 2] = k[1] * out_a2_common
    kernel_view[2, 0] = np.conjugate(kernel_view[0, 2])
    kernel_view[2, 1] = np.conjugate(kernel_view[1, 2])

    np.fill_diagonal(exp_cosh, exp_kDz)
    kernel_view[2, 2] = -exp_cosh
    return -kernel


def Demag_Kernel_K0(lz: int, dz: float,
                    dtype: str = 'float64') -> npt.NDArray:
    '''
    Returns the demagnetization kernel K[i,a,j,b](k=0)
    for a film discretized in lz parts, with height dz,
    and for a wavevector 0
    '''
    kernel = np.zeros((lz, 3, lz, 3), dtype=dtype)
    np.fill_diagonal(kernel[:, 2, :, 2], 1)
    return -kernel
