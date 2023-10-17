# Dymas micromagnetic software.
# Author: Diego González Chávez
# email : diegogch@cbpf.br

import numpy as np
import numpy.typing as npt


'''
Indices convention:
i,j,k or l,m,n are related to positons x,y,z  (not used anymore)  
x,y are related to positions as a list (flatten ijk indices with a mask)
a, b, c, d, e are related to vector component in 3d space  
u,v are realted to vector component in the plane locally orthogonal to m
'''

def torque_operator_LL(alpha: npt.NDArray,
                       gamma: npt.NDArray,
                       m: npt.NDArray) -> npt.NDArray:
    '''
    Toque operator in the LL equation
     - gamma/(1+alpha**2) m x [ ]
    '''
    return (-(gamma/(1+alpha**2))[..., None, None]
            * np.einsum('abc,xb->xac', LeviCivita, m))


def torque_operator_LLG(alpha: npt.NDArray,
                        gamma: npt.NDArray,
                        m: npt.NDArray) -> npt.NDArray:
    '''
    Toque operator in the LLG equation
     - gamma m x [ ]
    '''
    return - gamma[..., None, None] * np.einsum('abc,xb->xac',
                                                LeviCivita, m)


def damping_operator_LL(alpha: npt.NDArray,
                        gamma: npt.NDArray,
                        m: npt.NDArray) -> npt.NDArray:
    '''
    Damping operator in the LL equation
    gamma/(1+alpha**2) alpha m x [] x m
    '''
    return ((gamma * alpha/(1+alpha**2))[..., None, None]
            * (np.eye(3)[None,  :, :] -
               np.einsum('xa,xb->xab',  m, m)))


def damping_operator_LLG(alpha: npt.NDArray,
                         gamma: npt.NDArray,
                         m: npt.NDArray) -> npt.NDArray:
    '''
    Damping operator in the LLG equation
    damping_operator_LL + alpha**2 * torque_operator_LL + 
    '''
    return (damping_operator_LL(alpha, gamma, m)
            + alpha[..., None, None]**2*torque_operator_LL(alpha, gamma, m))


LeviCivita = np.array([[[0,  0,  0],
                        [0,  0,  1],
                        [0, -1,  0]],
                       [[0,  0, -1],
                        [0,  0,  0],
                        [1,  0,  0]],
                       [[0,  1,  0],
                        [-1,  0,  0],
                        [0,  0,  0]]])


def Zeeman_Kernel(H: npt.NDArray, m: npt.NDArray) -> npt.NDArray:
    return np.einsum('a,ijkb->aijkb', H, m)[:, :, :, :, :, None, None, None]


def D_operator(system):
    '''
    H_ef_eq[x,a] Effective field at equilibrium magnetization
    '''
    H_ef_eq = system.H + np.einsum('xayb,yb->xa',
                                   system.K_Total,
                                   system.m)
    '''
    N[x,a,y,b] = K - (H_ef_eq . m_eq) I perturbation magnetization to 
    perturbation field operator dh = N dm
    '''
    N = system.K_Total - np.einsum('ab,xy,x->xayb',
                                   np.eye(3),
                                   np.eye(system.m.shape[0]),
                                   np.einsum('xa,xa->x',
                                             H_ef_eq,
                                             system.m),
                                   optimize='greedy')
    '''
    P[x,a,b] projection operator to the plane locally orthogonal to m
    '''
    P = np.eye(3)[None, :, :] - np.einsum('xa,xb->xab',
                                          system.m,
                                          system.m)

    '''
    R[i,j,k,a,u] Operator that transform vectors from the basis 
    e[a] = {ex, ey, ez} to the basis e[u] = {e1,e2} 
    that is locally orthogonal to  m[x,a]
    '''
    R = np.linalg.eigh(P)[1][..., [1, 2]]

    '''
    L0[i,j,k,a,b] Operator from the LLG equation
    '''
    L0 = - np.einsum('xb,abc->xac',
                     system.m,
                     LeviCivita,
                     optimize='greedy')*system.gamma[..., None, None]

    '''
    L[i,j,k,a,b] Operator from the LLG equation
    '''
    L = (np.einsum('x,ab->xab',
                   system.alpha,
                   np.eye(3),
                   optimize='greedy')
         - np.einsum('xb,abc->xac',
                     system.m,
                     LeviCivita,
                     optimize='greedy')
         )*(system.gamma / (1+system.alpha**2))[..., None, None]

    '''
    D[x,l,y,v] Operator of the eigen-system
    i omega dm = D dm
    '''
    system.D0 = np.einsum('xab,xbc,xcyd->xayd',
                          L0, P, N, optimize='greedy')
    system.RD0R = np.einsum('xau,xayd,ydv->xuyv',
                            R, system.D0, R, optimize='greedy')

    system.D = np.einsum('xab,xbc,xcyd->xayd',
                         L, P, N, optimize='greedy')
    system.RDR = np.einsum('xau,xayd,ydv->xuyv',
                           R, system.D, R, optimize='greedy')
    mSize = system.m.shape[0]*2
    system.L0 = L0
    system.L = L
    system.P = P
    system.R = R
    system.N = N
    system.RD0Rz = system.RD0R.reshape(mSize, mSize)
    system.RDRz = system.RDR.reshape(mSize, mSize)


def T_operator(system):
    '''
    Indices convention:
    ijk or lmn are related to positons xyz
    a, b, c, d, e are related to vector component in 3d space
    u,v are realted to vector component in the plane locally orthogonal to m
    '''

    '''
    H_ef_eq[x,a] Effective field at equilibrium magnetization
    '''
    H_ef_eq = system.H + np.einsum('xayb,yb->xa',
                                   system.K_Total,
                                   system.m)
    '''
    N[x,a,y,b] = K - (H_ef_eq . m_eq) I perturbation magnetization to 
    perturbation field operator dh = N dm
    '''
    N = system.K_Total - np.einsum('ab,xy,x->xayb',
                                   np.eye(3),
                                   np.eye(system.m.shape[0]),
                                   np.einsum('xa,xa->x',
                                             H_ef_eq,
                                             system.m),
                                   optimize='greedy')
    '''
    Cr[x,a,b] cross product with m operator [m x]
    '''
    Cr = np.einsum('xb,abc->xac',
                   system.m,
                   LeviCivita)

    '''
    T[x,a,u] Operator that that transform vectors from the basis 
    e[a] = {ex, ey, ez} to the basis e[u] = {e1,e2} formed by the autovectors
    of i*Cr that are perpendicular to m_eq 
    '''
    T = np.linalg.eigh(1j*Cr)[1][..., [0, 2]]

    '''
    S representation of the [m_eq x] operator in the T base
    '''
    S = np.diag([-1, 1])

    '''
    D0 operator D0 = gamma S T.T.conj() N T
    '''
    D0 = np.einsum('xuv, xav, xayb, ybw -> xuyw',
                   1.0j * S[None, :, :] * system.gamma[..., None, None],
                   T.conj(),
                   N,
                   T,
                   optimize='greedy')

    '''
    D_alpha operator D_alpha = 1/(1+alpha**2) [I - i alpha S] D0
    '''
    D_alpha = np.einsum('xuv, xvyw -> xuyw',
                        (np.eye(2)[None, :, :]
                         - 1j * system.alpha[:, None, None] * S[None, :, :])
                        / (1+system.alpha**2)[..., None, None],
                        D0)

    mSize = system.m.shape[0]*2
    system.T = T
    system.TD0Tz = D0.reshape(mSize, mSize)
    system.TDTz = D_alpha.reshape(mSize, mSize)

