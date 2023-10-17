# Dymas micromagnetic software.
# Author: Diego González Chávez
# email : diegogch@cbpf.br

# Demagnetizing kernels for grids.

# References:
# A generalization of the Demagnetizing Tensor for Nonuniform Magnetization
# Newell, Williams and Dunlop, J.
# J. Geoph. Res. 98, 9551 (1993)

import numpy as np
import numpy.typing as npt
from typing import List, Union

## F precursors ##
# N_{xx} = FF(X,Y,Z,dX,dY,dZ)
# N_{yy} = FF(Y,Z,X,dY,dZ,dX)
# N_{zz} = FF(Z,X,Y,dZ,dX,dY)
#
# FF(A,B,C,dA,dB,dC) = [+ 2*F(A,B,C,dB,dC)
#                       - F(A+dA,B,C,dB,dC)
#                       - F(A-dA,B,C,dB,dC)] / 4*pi*dA*dB*dC
#
# F(A,B,C,dB,dC) = + F2(A, B+dB, C+dC)
#                  + F2(A, B-dB, C-dC)
#                  + F2(A, B-dB, C+dC)
#                  + F2(A, B+dB, C-dC)
#                  - 2*F2(A, B, C+dC)
#                  - 2*F2(A, B, C-dC)
#                  - 2*F2(A, B+dB, C)
#                  - 2*F2(A, B-dB, C))
#                  + 4*F2(A, B, C))
#
# F2(A,B,C) = 1/6 * {2*A**3
#                   + pB**3 + pC**3 - R**3
#                   + 3 * B * (C2 - A2) * np.arcsinh(B/pB)
#                   + 3 * C * (B2 - A2) * np.arcsinh(C/pC)
#                   + 3 * A2 * [B * np.arcsinh(B/A) +
#                               C * np.arcsinh(C/A)
#                               + R - pB - pC]
#                   - 6 * A*B*C * np.arctan((B*C)/(A*R))
#                   }
# F2(0,B,C) = 1/6 * [ B**3 + C**3 - pA**3
#                    + 3 * B * C**2 * arcsinh(B/C)
#                    + 3 * C * B**2 * arcsinh(C/B) ]
# F2(A,0,C) = F2(A,B,0) = 0
#
# R = (A**2 + B**2 + C**2)**0.5
# pA = (B**2 + C**2)**0.5
# pB = (A**2 + C**2)**0.5
# pC = (A**2 + B**2)**0.5


def _F2_A0(B: npt.NDArray, C: npt.NDArray) -> npt.NDArray:
    '''
    Special case of F2(A,B,C), calculated for A = 0.
     B and C should be positive
    '''
    pA = (B**2 + C**2)**0.5
    return (B**3 + C**3 - pA**3
            + 3 * B * C**2 * np.arcsinh(B/C)
            + 3 * C * B**2 * np.arcsinh(C/B)) / 6


def _F2(A: npt.NDArray, B: npt.NDArray,  C: npt.NDArray) -> npt.NDArray:
    '''
    Special case of F2(A,B,C), for A != 0 and B != 0 and C != 0.
     A, B and C should be positive
    '''
    A2 = A**2
    B2 = B**2
    C2 = C**2
    R = (A2 + B2 + C2)**0.5
    pB = (A2 + C2)**0.5
    pC = (A2 + B2)**0.5

    # Optimized algebra (first pass)
    return (+ 1/6 * (B2*pC + C2*pB - R**3)
            + 1/3 * A2 * (A - pC - pB)
            + 1/2 * (+ B * (C2 - A2) * np.arcsinh(B/pB)
                     + C * (B2 - A2) * np.arcsinh(C/pC)
                     + A2 * (B * np.arcsinh(B/A) +
                             C * np.arcsinh(C/A) + R)
                     )
            - A*B*C * np.arctan((B*C)/(A*R))
            )


def F2(A: npt.NDArray, B: npt.NDArray,  C: npt.NDArray) -> npt.NDArray:
    '''
    F2(A,B,C) precursor.
    '''
    A = np.abs(A.copy())
    B = np.abs(B.copy())
    C = np.abs(C.copy())

    output = np.zeros_like(A)
    eps = np.finfo(A.dtype).eps  # Epsilon value for rounding to zero
    mask_CB = (B < eps) + (C < eps)
    mask_A = (A < eps) * ~mask_CB  # _F2_A0 cases
    mask_other = ~(mask_A + mask_CB)  # _F2 cases

    # output[mask_CB] = 0
    if mask_A.any():
        output[mask_A] = _F2_A0(B[mask_A],
                                C[mask_A])
    if mask_other.any():
        output[mask_other] = _F2(A[mask_other],
                                 B[mask_other],
                                 C[mask_other])

    return output


def F(A: npt.NDArray, B: npt.NDArray,  C: npt.NDArray,
      dB: float,  dC: float) -> npt.NDArray:
    '''
    F(A, B, C, dB, dC) precursor.
    '''
    Bp = B+dB
    Bn = B-dB
    Cp = C+dC
    Cn = C-dC
    return (F2(A, Bp, Cp) + F2(A, Bn, Cn) + F2(A, Bn, Cp) + F2(A, Bp, Cn)
            - 2*(F2(A, B, Cp) + F2(A, B, Cn) + F2(A, Bp, C) + F2(A, Bn, C))
            + 4 * F2(A, B, C))


def FF(A: npt.NDArray, B: npt.NDArray,  C: npt.NDArray,
       dA: float, dB: float,  dC: float) -> npt.NDArray:
    '''
    FF(A, B, C, dA, dB, dC) precursor.
    '''
    return (2*F(A, B, C, dB, dC)
            - F(A+dA, B, C, dB, dC)
            - F(A-dA, B, C, dB, dC)) / (4*np.pi*dA*dB*dC)


## G precursors ##
# N_{xy} = GG(X,Y,Z,dX,dY,dZ)
# N_{yz} = GG(Y,Z,X,dY,dZ,dX)
# N_{zx} = GG(Z,X,Y,dZ,dX,dY)
#
# GG(A,B,C,dA,dB,dC) = [+ G(A   ,B   ,C,dA,dB,dC)
#                       - G(A-dA,B   ,C,dA,dB,dC)
#                       - G(A   ,B+dB,C,dA,dB,dC)
#                       + G(A-dA,B+dB,C,dA,dB,dC)] / 4*pi*dA*dB*dC
#
# G(A,B,C,dA,dB,dC) = + G2(A+dA,B,C+dC) - G2(A+dA,B-dB,C+dC)
#                     - G2(A,   B,C+dC) + G2(A,   B-dB,C+dC)
#                     + G2(A+dA,B,C-dC) - G2(A+dA,B-dB,C-dC)
#                     - G2(A,   B,C-dC) + G2(A,   B-dB,C-dC)
#                     + [-G2(A+dA,B,C) + G2(A+dA,B-dB,C)
#                        +G2(A   ,B,C) - G2(A,   B-dB,C)]*2
#
# G2(A,B,C) = 1/6 {  6*A*B*C*arcsinh(C/pC)
#                  + B*(3*C**2-B**2)*arcsinh(A/pA)
#                  + A*(3*C**2-A**2)*arcsinh(B/pB)
#                  - C**3*arctan[(A*B)/(C*R)]
#                  - 3*C*B**2*arctan[(A*C)/(B*R)]
#                  - 3*C*A**2*arctan[(B*C)/(A*R)]
#                  - 2*A*B*R
#                  + B**3*arcsinh(A/|B|)
#                  + A**3*arcsinh(B/|A|)
#                  + 2*A*B*pC
#                 }
# G2(0,B,C) = G2(A,0,C) = G2(A,B,0) = 0
#
# R = (A**2 + B**2 + C**2)**0.5
# pA = (B**2 + C**2)**0.5
# pB = (A**2 + C**2)**0.5
# pC = (A**2 + B**2)**0.5

def _G2(A: npt.NDArray, B: npt.NDArray,  C: npt.NDArray) -> npt.NDArray:
    '''
    Special case of G2(A,B,C), for A != 0 and B != 0 and C != 0.
    '''
    A2 = A**2
    B2 = B**2
    C2 = C**2
    R = (A2 + B2 + C2)**0.5
    pA = (B2 + C2)**0.5
    pB = (A2 + C2)**0.5
    pC = (A2 + B2)**0.5
    return (6 * A*B*C * np.arcsinh(C/pC)
            + B * (3*C2-B2) * np.arcsinh(A/pA)
            + A * (3*C2-A2) * np.arcsinh(B/pB)
            - C**3 * np.arctan((A*B)/(C*R))
            - 3 * C * B2 * np.arctan((A*C)/(B*R))
            - 3 * C * A2 * np.arctan((B*C)/(A*R))
            - 2 * A*B*R
            + B**3 * np.arcsinh(A/np.abs(B))
            + A**3 * np.arcsinh(B/np.abs(A))
            + 2 * A*B * pC) / 6


def G2(A: npt.NDArray, B: npt.NDArray,  C: npt.NDArray) -> npt.NDArray:
    '''
    G2(A,B,C) precursor.
    '''
    output = np.zeros_like(A)
    eps = 1e-15  # Epsilon value for rounding to zero
    mask_zeros = (np.abs(A) < eps) + (np.abs(B) < eps) + (np.abs(C) < eps)

    # output[mask_zeros] = 0
    output[~mask_zeros] = _G2(A[~mask_zeros], B[~mask_zeros], C[~mask_zeros])
    return output


def G(A: npt.NDArray, B: npt.NDArray,  C: npt.NDArray,
      dA: float, dB: float,  dC: float) -> npt.NDArray:
    '''
    G(A, B, C, dA, dB, dC) precursor.
    '''
    Ap = A+dA
    Bn = B-dB
    Cp = C+dC
    Cn = C-dC
    return (+ G2(Ap, B, Cp) - G2(Ap, Bn, Cp) - G2(A, B, Cp) + G2(A, Bn, Cp)
            + G2(Ap, B, Cn) - G2(Ap, Bn, Cn) - G2(A, B, Cn) + G2(A, Bn, Cn)
            + 2*(G2(A, B, C) - G2(A, Bn, C) - G2(Ap, B, C) + G2(Ap, Bn, C)))


def GG(A: npt.NDArray, B: npt.NDArray,  C: npt.NDArray,
       dA: float, dB: float,  dC: float) -> npt.NDArray:
    '''
    GG(A, B, C, dA, dB, dC) precursor.
    '''
    An = A-dA
    Bp = B+dB
    return (G(A, B, C, dA, dB, dC)
            - G(An, B, C, dA, dB, dC)
            - G(A, Bp, C, dA, dB, dC)
            + G(An, Bp, C, dA, dB, dC)) / (4*np.pi*dA*dB*dC)


def Kernel_Values(lx: int, ly: int, lz: int,
                  dx: float, dy: float, dz: float,
                  dtype: str = 'float64',
                  copies: npt.NDArray = None) -> npt.NDArray:
    '''
    Returns the demagnetization kernel values 
    for a grid of shape (lx, ly, lz) and cell size dx*dy*dz
    #copies = position list of copies (used for PBC)
    The resulting array is of shape (6,lx,ly,lz)
    The array contains the values
    [Nxx, Nyy, Nzz, Nxy, Nxz, Nyz] of the interaction of 
    all the cells with the cell at (0,0,0)
    '''
    dXYZ = np.array([dx, dy, dz], dtype=dtype)
    ijk = np.mgrid[0:lx, 0:ly, 0:lz]

    scale = dXYZ.min()
    dX, dY, dZ = dXYZ/scale
    X, Y, Z = ijk*dXYZ[:, None, None, None]/scale

    output = np.zeros((6, *X.shape), dtype=dtype)
    Nxx, Nyy, Nzz, Nxy, Nxz, Nyz = output

    if copies is None:
        copies = [[0, 0, 0]]

    for offset in copies/scale:
        Xo = X + offset[0]
        Yo = Y + offset[1]
        Zo = Z + offset[2]

        Nxx -= FF(Xo, Yo, Zo, dX, dY, dZ)
        Nyy -= FF(Yo, Zo, Xo, dY, dZ, dX)
        Nzz -= FF(Zo, Xo, Yo, dZ, dX, dY)

        Nxy -= GG(Xo, Yo, Zo, dX, dY, dZ)
        Nxz -= GG(Xo, Zo, Yo, dX, dZ, dY)
        Nyz -= GG(Yo, Zo, Xo, dY, dZ, dX)

    return output


def Expanded_Kernel_Values(lx: int, ly: int, lz: int,
                           dx: float, dy: float, dz: float,
                           dtype: str = 'float64',
                           copies: npt.NDArray = None) -> npt.NDArray:
    '''
    Returns the demagnetization kernel values 
    for a grid of shape (lx, ly, lz) and cell size dx*dy*dz
    The resulting array is of shape (3,3,2*lx-1,2*ly-1,2*lz-1)
    The array contains the values
    [Nxx, Nxy, Nxz, 
     Nxy, Nyy, Nyz,
     Nxz, Nyz, Nzz] of the interaction of all the cells 
     with the cell at (lx-1,ly-1,lz-1) which is in the center of the
     new grid (2*lx-1, 2*ly-1, 2*lz-1)
    '''
    values = Kernel_Values(lx, ly, lz, dx, dy, dz, dtype, copies)

    # indices for the 3x3 matrix
    li = np.tril_indices(3, -1)  # lower triangle
    di = np.diag_indices(3)  # diagonal
    ui = np.triu_indices(3, +1)  # upper triangle

    # allocate space for the expanded values
    kernel = np.zeros((3, 3, 2*lx-1, 2*ly-1, 2*lz-1))

    # Fill the "XYZ positive + 0" volume
    region_0 = kernel[:, :, lx-1:, ly-1:, lz-1:]
    region_0[di] = values[:3]  # Nxx, Nyy, Nzz
    region_0[ui] = region_0[li] = values[3:]  # Nxy, Nxz, Nyz

    # Positive and negative slices for each axis
    pX = slice(lx, None)
    pY = slice(ly, None)
    pZ = slice(lz, None)
    nX = slice(None, lx-1)
    nY = slice(None, ly-1)
    nZ = slice(None, lz-1)

    # Parities for mirror transformations
    parityX = np.array([[+1, -1, -1],
                        [-1, +1, +1],
                        [-1, +1, +1]])[:, :, None, None, None]
    parityY = np.array([[+1, -1, +1],
                        [-1, +1, -1],
                        [+1, -1, +1]])[:, :, None, None, None]
    parityZ = np.array([[+1, +1, -1],
                        [+1, +1, -1],
                        [-1, -1, +1]])[:, :, None, None, None]

    # Mirror operations
    kernel[..., nX, :, :] = kernel[..., pX, :, :][..., ::-1, :, :]*parityX
    kernel[..., nY, :] = kernel[..., pY, :][..., ::-1, :]*parityY
    kernel[..., nZ] = kernel[..., pZ][..., ::-1]*parityZ

    return kernel


def Demag_Kernel(lx: int, ly: int, lz: int,
                 dx: float, dy: float, dz: float,
                 dtype: str = 'float64',
                 copies: npt.NDArray = None) -> npt.NDArray:
    '''
    Returns the demagnetization kernel K[i,j,k,a,l,m,n,b]
    for a grid of shape (lx, ly, lz) and cell size dx*dy*dz
    The demag field Hd = N.m is calculated by
    Hd[i,j,k,a] = einsum('ijkalmnb,lmnb->ijka', kernel, m)
    where kernel[i,j,k,:,l,m,n,:] is the matrix Demagnetization factor of 
    the cell at position (i,j,k) with the cell at position (l,m,n)
    '''
    # Get the values
    values = Expanded_Kernel_Values(lx, ly, lz, dx, dy, dz, dtype, copies)
    # allocate memory
    kernel = np.empty((lx, ly, lz, 3, lx, ly, lz, 3), dtype=dtype)
    kernel_view = np.einsum('ijkalmnb->abijklmn', kernel)
    # fill up the values
    with np.nditer(np.zeros((lx, ly, lz)), flags=['multi_index']) as it:
        for x in it:
            i, j, k = it.multi_index
            kernel_view[:, :, i, j, k] = values[:, :,
                                                lx-1-i:2*lx-1-i,
                                                ly-1-j:2*ly-1-j,
                                                lz-1-k:2*lz-1-k]
    return kernel


def Mesh(lx: int, ly: int, lz: int,
         dx: float, dy: float, dz: float) -> npt.NDArray:
    '''
    Returns the coodinate positions Mesh[a,i,j,k]
    for the center of each cell
    '''
    xs = np.arange(0, lx, 1)*dx + dx/2
    ys = np.arange(0, ly, 1)*dy + dy/2
    zs = np.arange(0, lz, 1)*dz + dz/2
    return np.array(np.meshgrid(xs, ys, zs, indexing='ij'))


def MagneticDipoleField(m: Union[npt.NDArray, List[float]],
                        xyz: npt.NDArray) -> npt.NDArray:
    '''
    Calculates the field of a magnetic dipole m
    at positions xyz
    '''
    m = np.atleast_1d(m)
    rabs = np.linalg.norm(xyz, axis=0)+1E-100
    r_hat = np.atleast_1d(xyz)/rabs
    s = [np.newaxis for _ in r_hat.shape]
    s[0] = slice(None)
    s = tuple(s)
    return 1.0E-7 * (3*r_hat*np.einsum('m,m...->...', m, r_hat) - m[s])/rabs**3


def CopiesPositions(repetitions: List[int],
                    offsets: List[float]) -> npt.NDArray:
    """_summary_
    Used to calculate the postion of the copies of PBCs

    Args:
        repetitions (List[int]): Number of repetitions for each axis
        offsets (List[float]): Offsets for each axis.

    Returns:
        npt.NDArray: Positions of the repetitions
    """

    both_sides_repetitions = np.array(repetitions)*2 + 1
    q = Mesh(*both_sides_repetitions, *offsets)
    center = both_sides_repetitions * np.array(offsets) / 2
    positions = q.reshape((3, q.size//3)).T - center
    # Return without [0,0,0]
    return np.delete(positions, positions.shape[0]//2, axis=0)
