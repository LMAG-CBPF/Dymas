# Demagnetizing tensor calculations, using plane waves

import numpy as np
import numpy.typing as npt


def Demag_Kernel_K(lz: int, dz: float, k: npt.NDArray,
                   dtype: str = 'complex128') -> npt.NDArray:
    '''
    Returns the demagnetization kernel K[i,a,j,b](k)
    for an infinite film discretized in lz parts, with height dz,
    and for a wavevector k
    '''
    k_theta = np.arctan2(k[1], k[0])
    abs_k = np.linalg.norm(k)
    kDz = abs_k*dz

    # Calculate reusable terms
    sinc_kdz = np.real(np.sinc(1.0j*kDz/(2*np.pi)))
    exp_kdz = np.exp(-kDz/2)

    grid = np.mgrid[0:lz]
    Zi_Zj = dz*(grid[None, :] - grid[:, None])
    exp_kzizj = np.exp(-abs_k*np.abs(Zi_Zj))  # This is a matrix [ij]

    # allocate memory
    kernel = np.zeros((lz, 3, lz, 3), dtype=dtype)
    kernel_view = np.einsum('iajb->abij', kernel)

    # fill up the values

    # Case a,b != 2 (x_a, x_b != z)
    out_ab = -exp_kzizj * np.sinh(kDz/2) * sinc_kdz
    np.fill_diagonal(out_ab, -(1-exp_kdz*sinc_kdz))  # i=j
    kernel_view[0, 0] = np.cos(k_theta)**2 * out_ab
    kernel_view[0, 1] = np.cos(k_theta) * np.sin(k_theta) * out_ab
    kernel_view[1, 0] = np.sin(k_theta) * np.cos(k_theta) * out_ab
    kernel_view[1, 1] = np.sin(k_theta)**2 * out_ab
    # Note: Since out_ab == 0 for abs_k = 0 
    # this equations provide the correct results for k=[0,0]

    # Case a xor b = 2
    out_a2 = 1j * np.sign(Zi_Zj) * exp_kzizj * np.sinh(kDz/2) * sinc_kdz
    np.fill_diagonal(out_a2, 0)  # i=j
    kernel_view[0, 2] = kernel_view[2, 0] = np.cos(k_theta) * out_a2
    kernel_view[1, 2] = kernel_view[2, 1] = np.sin(k_theta) * out_a2
    # Note: Since out_a2 == 0 for abs_k = 0 
    # this equations provide the correct results for k=[0,0]

    # Case a and b = 2
    out_22 = exp_kzizj * np.sinh(kDz/2) * sinc_kdz
    np.fill_diagonal(out_22, -exp_kdz*sinc_kdz)  # i=j
    kernel_view[2, 2] = out_22

    return kernel


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
