# Dymas micromagnetic software.
# Author: Diego González Chávez
# email : diegogch@cbpf.br

# Exchange and DMI kernels for grids.

import numpy as np
import numpy.typing as npt
from typing import List

LeviCivita = np.array([[[0.,  0.,  0.],
                        [0.,  0.,  1.],
                        [0., -1.,  0.]],
                       [[0.,  0., -1.],
                        [0.,  0.,  0.],
                        [1.,  0.,  0.]],
                       [[0.,  1.,  0.],
                        [-1.,  0.,  0.],
                        [0.,  0.,  0.]]], dtype='int8')


def get_edges(mask: npt.NDArray) -> List[npt.NDArray]:
    '''
    Retuns the edges of the mask of a mesh.
    The output is a list with 6 elements:
    [positive_X_edges, negative_X_edges,
     positive_Y_edges, negative_Y_edges,
     positive_Z_edges, negative_Z_edges]
    where each element of the list is 
    a boolean array with same shape of the mask
    '''
    lx, ly, lz = mask.shape
    # extended mask
    ex_mask = np.zeros((lx+2, ly+2, lz+2))
    ex_mask[1:-1, 1:-1, 1:-1] = mask

    edges = []
    edges.append((mask - ex_mask[0:-2, 1:-1, 1:-1]) == 1)  # X possitive
    edges.append((mask - ex_mask[2:, 1:-1, 1:-1]) == 1)  # X negative
    edges.append((mask - ex_mask[1:-1, 0:-2, 1:-1]) == 1)  # Y possitive
    edges.append((mask - ex_mask[1:-1, 2:, 1:-1]) == 1)  # Y negative
    edges.append((mask - ex_mask[1:-1, 1:-1, 0:-2]) == 1)  # Z possitive
    edges.append((mask - ex_mask[1:-1, 1:-1, 2:]) == 1)  # Z negative
    return edges


def Exchange_Kernel(lx: int, ly: int, lz: int,
                    dx: float, dy: float, dz: float,
                    A_eff: npt.NDArray) -> npt.NDArray:
    """_summary_

    Args:
        lx (int): _description_
        ly (int): _description_
        lz (int): _description_
        dx (float): _description_
        dy (float): _description_
        dz (float): _description_
        A_eff (npt.NDArray): _description_

    Returns:
        npt.NDArray: _description_
    """
    '''
    Returns the Exchange kernel
    for a grid of shape (lx, ly, lz) and cell size dx*dy*dz
    The exchange field Hex = Kex.m is calculated by
    Hex = A*einsum('aijklmnb,lmnb->ijka', kernel_ex, m)
    where kernel_ex[i,j,k,a,l,m,n,b] is the matrix exchange interaction of
    the cell at position (i,j,k) with the cell at position (l,m,n)
    '''
    # define a multidimension Kronecker delta
    kd = np.einsum('il,jm,kn->ijklmn',
                   np.eye(lx+2, dtype='int8')[:, 1:-1],
                   np.eye(ly+2, dtype='int8')[:, 1:-1],
                   np.eye(lz+2, dtype='int8')[:, 1:-1])

    dx2i = 1/np.array([dx, dy, dz])**2
    # Calculate the (d2/d2x + d2/d2y + d2/d2z) discrete operator
    # d2/d2x.m[ijk] = (m[i-1,j,k] - 2m[ijk] + m[i+1,j,k])/dx**2
    #    = [(kd[i-1,j,k,lmn] - 2*kd[i,j,k,lmn] + kd[i+1,j,k,lmn])/dx**2].m[lmn]
    K_exh = (+ (2*kd[1:-1, 1:-1, 1:-1]
             - kd[0:-2, 1:-1, 1:-1]
             - kd[2:, 1:-1, 1:-1])*dx2i[0]
             + (2*kd[1:-1, 1:-1, 1:-1]
             - kd[1:-1, 0:-2, 1:-1]
             - kd[1:-1, 2:, 1:-1])*dx2i[1]
             + (2*kd[1:-1, 1:-1, 1:-1]
             - kd[1:-1, 1:-1, 0:-2]
             - kd[1:-1, 1:-1, 2:])*dx2i[2])
    mask = (A_eff == 0)
    K_exh[mask, ...] = 0
    edges = get_edges(~mask)
    for edge, dxi in zip(edges, np.c_[dx2i, dx2i].flatten()):
        K_exh[edge, edge] -= dxi
    return np.einsum('lmn,ijklmn,ab->ijkalmnb',
                     A_eff,
                     K_exh,
                     np.eye(3),
                     optimize=True)


def DMI_Kernel(lx: int, ly: int, lz: int,
               dx: float, dy: float, dz: float,
               D_eff: npt.NDArray) -> npt.NDArray:
    '''
    Returns the Exchange kernel
    for a grid of shape (lx, ly, lz) and cell size dx*dy*dz
    The exchange field Hex = Kex.m is calculated by
    Hex = A*einsum('aijklmnb,lmnb->ijka', kernel_ex, m)
    where kernel_ex[a,b,i,j,k,l,m,n] is the matrix exchange interaction of
    the cell at position (i,j,k) with the cell at position (l,m,n)
    '''
    # define a multidimension Kronecker delta
    kd = np.einsum('il,jm,kn->ijklmn',
                   np.eye(lx+2, dtype='int8')[:, 1:-1],
                   np.eye(ly+2, dtype='int8')[:, 1:-1],
                   np.eye(lz+2, dtype='int8')[:, 1:-1])
    d2xi = 1/(np.array([dx, dy, dz])*2)

    K_grad = np.array([(kd[2:, 1:-1, 1:-1] - kd[0:-2, 1:-1, 1:-1])*d2xi[0],
                       (kd[1:-1, 2:, 1:-1] - kd[1:-1, 0:-2, 1:-1])*d2xi[1],
                       (kd[1:-1, 1:-1, 2:] - kd[1:-1, 1:-1, 0:-2])*d2xi[2]])
    mask = (D_eff == 0)
    K_grad[:, mask, ...] = 0
    edges = get_edges(~mask)
    K_grad[0, edges[0], edges[0]] -= d2xi[0]
    K_grad[0, edges[1], edges[1]] += d2xi[0]
    K_grad[1, edges[2], edges[2]] -= d2xi[1]
    K_grad[1, edges[3], edges[3]] += d2xi[1]
    K_grad[2, edges[4], edges[4]] -= d2xi[2]
    K_grad[2, edges[5], edges[5]] += d2xi[2]
    return np.einsum('lmn,bijklmn,abc->ijkalmnb',
                     D_eff,
                     K_grad,
                     LeviCivita,
                     optimize=True)
