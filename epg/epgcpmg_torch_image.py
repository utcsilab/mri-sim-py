#!/usr/bin/python

# EPG CPMG simulation code, based off of Matlab scripts from Brian Hargreaves <bah@stanford.edu> 
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
# 2019 Ke Wang <kewang@eecs.berkeley.edu> rewrite it in Pytorch

from __future__ import division
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from warnings import warn


def rf(FpFmZ, alpha):
    "Same as rf2, but only returns FpFmZ"""
    return rf2(FpFmZ, alpha)[0]
def rf2(FpFmZ, alpha):
    
    """ Propagate EPG states through an RF rotation of 
    alpha (radians). Assumes CPMG condition, i.e.
    magnetization lies on the real x axis.
    
    INPUT: (input should be a torch tensor)
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians

    OUTPUT:
        FpFmZ = Updated FpFmZ state.
        RR = RF rotation matrix (3x3).

    """
    if torch.abs(alpha) > 2*np.pi:
        warn('rf2: Flip angle should be in radians! alpha=%f' % alpha)
    cosa2 = torch.cos(alpha/2.)**2
    sina2 = torch.sin(alpha/2.)**2
    cosa = torch.cos(alpha)
    sina = torch.sin(alpha)
    RR = torch.tensor([ [cosa2, sina2, sina],
                    [sina2, cosa2, -sina],
                    [-0.5 * sina, 0.5 * sina, cosa] ]).cuda()
#     RR = torch.tensor([ [cosa2, sina2, sina],
#                     [sina2, cosa2, -sina],
#                     [-0.5 * sina, 0.5 * sina, cosa] ]).cuda()
    FpFmZ = torch.mm(RR,FpFmZ.float()) # dot in numpy
    return FpFmZ, RR

def rf_ex(FpFmZ, alpha):
    "Same as rf2_ex, but only returns FpFmZ"""
    return rf2_ex(FpFmZ, alpha)[0]
def rf2_ex(FpFmZ, alpha):
    """ Propagate EPG states through an RF excitation of 
    alpha (radians) along the y direction, i.e. phase of pi/2.
    in Pytorch
    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians

    OUTPUT:
        FpFmZ = Updated FpFmZ state.
        RR = RF rotation matrix (3x3).

    """

    try:
        alpha = alpha[0]
    except:
        pass


    if torch.abs(alpha) > 2 * np.pi:
        warn('rf2_ex: Flip angle should be in radians! alpha=%f' % alpha)

    cosa2 = torch.cos(alpha/2.)**2
    sina2 = torch.sin(alpha/2.)**2

    cosa = torch.cos(alpha)
    sina = torch.sin(alpha)
#     RR = torch.tensor([ [cosa2, -sina2, sina],
#                     [-sina2, cosa2, sina],
#                     [-0.5 * sina, -0.5 * sina, cosa] ])
    RR = torch.tensor([ [cosa2, -sina2, sina],
                    [-sina2, cosa2, sina],
                    [-0.5 * sina, -0.5 * sina, cosa] ]).cuda()
#     print(FpFmZ)
    FpFmZ = torch.mm(RR, FpFmZ)

    return FpFmZ, RR

def rf_prime(FpFmZ, alpha):
    """Same as rf_prime2, but only returns FpFmZ"""
    return rf_prime2(FpFmZ, alpha)[0]
def rf_prime2(FpFmZ, alpha):
    """ Compute the gradient of the RF rotation operator, where
    alpha (radians) is the RF rotation. Assumes CPMG condition, i.e.
    magnetization lies on the real x axis.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians

    OUTPUT:
        FpFmZ = Derivative of FpFmZ state w.r.t. alpha
        RR = Derivative of RF rotation matrix (3x3) w.r.t. alpha

    """

    if torch.abs(alpha) > 2 * np.pi:
        warn('rf_prime2: Flip angle should be in radians! alpha=%f' % alpha)

    RR = torch.tensor([ [-torch.cos(alpha/2.) * torch.sin(alpha/2.), torch.cos(alpha/2.) * torch.sin(alpha/2.), torch.cos(alpha)],
                    [torch.cos(alpha/2.) * torch.sin(alpha/2.), -torch.cos(alpha/2.) * torch.sin(alpha/2.), -torch.cos(alpha)],
                    [-0.5 * torch.cos(alpha), 0.5 * torch.cos(alpha), -torch.sin(alpha)] ])

    FpFmZ = torch.mm(RR, FpFmZ)

    return FpFmZ, RR


def rf_B1_prime(FpFmZ, alpha, B1):
    """Same as rf_B1_prime2, but only returns FpFmZ"""
    return rf_B1_prime2(FpFmZ, alpha, B1)[0]

def rf_B1_prime2(FpFmZ, alpha, B1):
    """ Compute the gradient of B1 inhomogeneity w.r.t. RF refocusing operator, where
    alpha (radians) is the RF rotation and B1 is the B1 homogeneity (0, 2).
    Assumes CPMG condition, i.e. magnetization lies on the real x axis.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians
        B1 = B1 Homogeneity, where 1. is homogeneous

    OUTPUT:
        FpFmZ = Derivative of FpFmZ state w.r.t. alpha
        RR = Derivative of RF rotation matrix (3x3) w.r.t. B1

    """

    if torch.abs(alpha) > 2 * np.pi:
        warn('rf_B1_prime2: Flip angle should be in radians! alpha=%f' % alpha)

    if B1 < 0 or B1 > 2:
        warn('rf_B1_prime2: B1 Homogeneity should be a percentage between (0, 2)')

    RR = torch.tensor([ [-alpha*torch.cos(B1*alpha/2.) * torch.sin(B1*alpha/2.), alpha*torch.cos(B1*alpha/2.) * torch.sin(B1*alpha/2.), alpha*torch.cos(B1*alpha)],
                    [alpha*torch.cos(B1*alpha/2.) * torch.sin(B1*alpha/2.), -alpha*torch.cos(B1*alpha/2.) * torch.sin(B1*alpha/2.), -alpha*torch.cos(B1*alpha)],
                    [-0.5*alpha*torch.cos(B1*alpha), 0.5*alpha*torch.cos(B1*alpha), -alpha*torch.sin(B1*alpha)] ])

    FpFmZ = torch.mm(RR, FpFmZ)

    return FpFmZ, RR


def rf_ex_B1_prime(FpFmZ, alpha, B1):
    """Gradient of B1 inhomogeneity w.r.t. RF excitation operator, where
    alpha (radians) is the RF rotation and B1 is the B1 honogeneity (0, 2).
    Assumes CPMG condition, i.e. RF excitation in the y direction.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians
        B1 = B1 Homogeneity, where 1. is homogeneous

    OUTPUT:
        FpFmZ = Derivative of FpFmZ state w.r.t. alpha
    """

    if torch.abs(alpha) > 2 * np.pi:
        warn('rf_ex_B1_prime2: Flip angle should be in radians! alpha=%f' % alpha)

    if B1 < 0 or B1 > 2:
        warn('rf_ex_B1_prime: B1 Homogeneity should be a percentage between (0, 2)')

    RR = torch.tensor([ [-alpha*torch.cos(B1*alpha/2.) * torch.sin(B1*alpha/2.), alpha*torch.cos(B1*alpha/2.) * torch.sin(B1*alpha/2.), alpha*torch.cos(B1*alpha)],
                    [alpha*torch.cos(B1*alpha/2.) * torch.sin(B1*alpha/2.), -alpha*torch.cos(B1*alpha/2.) * torch.sin(B1*alpha/2.), alpha*torch.cos(B1*alpha)],
                    [-0.5*alpha*torch.cos(B1*alpha), -0.5*alpha*torch.cos(B1*alpha), -alpha*torch.sin(B1*alpha)] ])

    FpFmZ = torch.tensor(RR, FpFmZ)

    return FpFmZ

def relax_mat(T, T1, T2):
    E2 = torch.exp(-T/T2)
    E1 = torch.exp(-T/T1)

    EE = torch.diag(torch.tensor([E2, E2, E1]))      # Decay of states due to relaxation alone.

    return EE

def relax_mat_prime_T1(T, T1, T2):
    E1_prime_T1 = T * torch.exp(-T/T1) / T1**2
    return torch.diag(torch.tensor([0, 0, E1_prime_T1]))

def relax_mat_prime_T2(T, T1, T2):
    E2_prime_T2 = T * torch.exp(-T/T2) / T2**2
    return torch.diag(torch.tensor([E2_prime_T2, E2_prime_T2, 0]))


def relax_prime_T1(FpFmZ, T, T1, T2):
    """returns E'(T1) FpFmZ + E0'(T1)"""
    
    EE_prime_T1 = relax_mat_prime_T1(T, T1, T2)
    
    RR = -EE_prime_T1[2,2]
    
    FpFmZ = torch.mm(EE_prime_T1, FpFmZ)
    FpFmZ[2,0] = FpFmZ[2,0] + RR
    
    return FpFmZ
def relax_prime_T2(FpFmZ, T, T1, T2):
    """returns E'(T2) FpFmZ"""
    
    EE_prime_T2 = relax_mat_prime_T2(T, T1, T2)
    FpFmZ = torch.mm(EE_prime_T2, FpFmZ)
    
    return FpFmZ


def relax(FpFmZ, T, T1, T2):
    """Same as relax2, but only returns FpFmZ"""
    return relax2(FpFmZ, T, T1, T2)[0]

def relax2(FpFmZ, T, T1, T2):
    """ Propagate EPG states through a period of relaxation over
    an interval T.
    torch
    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        T1, T2 = Relaxation times (same as T)
        T = Time interval (same as T1,T2)

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.
        EE = decay matrix, 3x3 = diag([E2 E2 E1]);

   """
#     n_t = torch.sqrt(FpFmZ.shape[1])
    E2 = torch.exp(-T/T2)
    E1 = torch.exp(-T/T1)
    E = torch.stack((E1,E1,E2)).transpose(1,0).flatten().float()
    FpFm = FpFmZ.transpose(1,0).flatten().float()
    FpFmZ2 = (FpFm*E).reshape(320*320,3).transpose(1,0)
    
#     EE = torch.diag(torch.tensor([E2, E2, E1])).cuda()      # Decay of states due to relaxation alone.
    
    
    
#     EE = torch.diag(torch.tensor([E2, E2, E1]))      # Decay of states due to relaxation alone.
    RR = 1 - E1                     # Mz Recovery, affects only Z0 state, as 
                                    # recovered magnetization is not dephased.

#     FpFmZ = torch.mm(EE, FpFmZ.double())       # Apply Relaxation
#     FpFmz = torch.DoubleTensor(FpFmz)
#     FpFmZ = torch.mm(EE, FpFmZ.double().cuda())       # Apply Relaxation
    FpFmZ2[2,:] = FpFmZ2[2,:] + RR.float()    # Recovery  

    return FpFmZ2, E

def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def grad(FpFmZ, noadd=False):
    """Propagate EPG states through a "unit" gradient. Assumes CPMG condition,
    i.e. all states are real-valued.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!

    OUTPUT:
        Updated FpFmZ state.

    """

    # Gradient does not affect the Z states.

    if noadd == False:
#         print(FpFmZ)
#         FpFmZ = torch.cat((FpFmZ.float().cuda(), torch.tensor([[0.],[0.],[0.]]).cuda()),1)   # add higher dephased state
        
        FpFmZ = torch.cat((FpFmZ.float(), torch.tensor([[0.],[0.],[0.]])),1)   # add higher dephased state

    FpFmZ[0,:] = roll(FpFmZ[0,:], 1,0)     # shift Fp states
    FpFmZ[1,:] = roll(FpFmZ[1,:], -1,0)    # shift Fm states
    FpFmZ[1,-1] = 0                         # Zero highest Fm state
    FpFmZ[0,0] = FpFmZ[1,0]                 # Fill in lowest Fp state

    return FpFmZ



def grad_2(FpFmZ):
    FpFmZ1 = FpFmZ.clone()
    FpFmZ[0,:] = 0
    FpFmZ[1,:] = 0
    FpFmZ1[1,:] = 0
    FpFmZ1[2,:] = 0
    return torch.cat((FpFmZ,FpFmZ1),1)

def grad_3(FpFmZ):
    FpFmZ1 = FpFmZ[:,:320*320]
    FpFmZ2 = FpFmZ[:,320*320:]
    FpFmZ_1 = torch.zeros(FpFmZ1.shape).cuda()
    FpFmZ_2 = torch.zeros(FpFmZ1.shape).cuda()
    FpFmZ_3 = torch.zeros(FpFmZ1.shape).cuda()
    FpFmZ_1[0,:] = FpFmZ2[1,:]
    FpFmZ_1[1,:] = FpFmZ2[1,:]
    FpFmZ_1[2,:] = FpFmZ1[2,:]
    FpFmZ_2[0,:] = FpFmZ1[0,:]
    FpFmZ_2[2,:] = FpFmZ2[2,:]
    FpFmZ_3[0,:] = FpFmZ2[0,:]
    
    return FpFmZ_1

def grad_simple(FpFmZ):
    FpFmZ[0,:] = 0
    FpFmZ[1,:] = 0
    return FpFmZ


def FSE_TE(FpFmZ, alpha, TE, T1, T2, noadd=False, recovery=True):
    """ Propagate EPG states through a full TE, i.e.
    relax -> grad -> rf -> grad -> relax.
    Assumes CPMG condition, i.e. all states are real-valued.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians
        T1, T2 = Relaxation times (same as TE)
        TE = Echo Time interval (same as T1, T2)
        noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.

   """

#     EE = relax_mat(TE/2., T1, T2)

    if recovery:
#         print(FpFmZ.dtype)
#         F1 = torch.DoubleTensor(FpFmZ)
        FpFmZ = relax(FpFmZ.double(), TE/2., T1, T2)
    else:
        FpFmZ = torch.mm(EE, FpFmZ)
#     P_grad = grad_2(FpFmZ)
#     print(FpFmZ)
    FpFmZ = grad_2(FpFmZ)
#     print(FpFmZ)
#     FpFmZ1 = P_grad[1]
    FpFmZ = rf(FpFmZ, alpha)
    FpFmZ = grad_3(FpFmZ)
    if recovery:
        FpFmZ = relax(FpFmZ, TE/2., T1, T2)
#         print(FpFmZ)
        
    else:
        FpFmZ = torch.mm(EE, FpFmZ)

    return FpFmZ


def FSE_TE_prime_alpha(FpFmZ, alpha, TE, T1, T2, noadd=False, recovery=True):
    """ Gradient of EPG over a full TE, w.r.t. flip angle alpha, i.e.
    relax -> grad -> rf_prime -> grad -> relax_hat,
    where rf_prime is the derivative of the RF pulse matrix w.r.t. alpha,
    and relax_hat  is the relaxation without longitudinal recovery
    Assumes CPMG condition, i.e. all states are real-valued.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians
        T1, T2 = Relaxation times (same as TE)
        TE = Echo Time interval (same as T1, T2)
        noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!
        recovery = True to include T1 recovery in the Z0 state.

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.

   """

    FpFmZ, EE = relax2(FpFmZ, TE/2., T1, T2)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf_prime(FpFmZ, alpha)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = torch.mm(EE, FpFmZ)

    return FpFmZ


def FSE_TE_prime1_T2(FpFmZ, alpha, TE, T1, T2, noadd=False):
    """ Returns E(T2) G R G E'(T2) FpFmZ"""
    
    EE = relax_mat(TE/2., T1, T2)
    EE_prime = relax_mat_prime_T2(TE/2., T1, T2)

    FpFmZ = torch.mm(EE_prime, FpFmZ)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf(FpFmZ, alpha)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = torch.mm(EE, FpFmZ)
    
    return FpFmZ


def FSE_TE_prime2_T2(FpFmZ, alpha, TE, T1, T2, noadd=False):
    """ Returns E'(T2) G R G (E(T2) FpFmZ + E0)"""
    
    EE_prime = relax_mat_prime_T2(TE/2., T1, T2)
    
    FpFmZ = relax(FpFmZ, TE/2., T1, T2)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf(FpFmZ, alpha)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = torch.mm(EE_prime, FpFmZ)
    
    return FpFmZ

def FSE_TE_prime1_T1(FpFmZ, alpha, TE, T1, T2, noadd=False):
    """ Returns E(T1) G R G (E'(T1) FpFmZ + E0'(T1))"""
    
    EE = relax_mat(TE/2., T1, T2)
    
    FpFmZ = relax_prime_T1(FpFmZ, TE/2., T1, T2) # E'(T1) FpFmZ + E0'(T1)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf(FpFmZ, alpha)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = torch.mm(EE, FpFmZ)
    
    return FpFmZ

def FSE_TE_prime2_T1(FpFmZ, alpha, TE, T1, T2, noadd=False):
    """ Returns E'(T1) G R G E(T1) FpFmZ + E0'(T1)"""
    
    EE = relax_mat(TE/2., T1, T2)

    FpFmZ = torch.mm(EE, FpFmZ) 
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf(FpFmZ, alpha)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = relax_prime_T1(FpFmZ, TE/2., T1, T2) # E'(T1) FpFmZ + E0'(T1)
    
    return FpFmZ


def FSE_TE_prime_B1(FpFmZ, alpha, TE, T1, T2, B1, noadd=False):
    """ Gradient of EPG over a full TE, w.r.t. B1 homogeneity fraciton B1, i.e.
    relax -> grad -> rf_B1_prime -> grad -> relax_hat,
    where rf_B1_prime is the derivative of the RF pulse matrix w.r.t. B1,
    and relax_hat  is the relaxation without longitudinal recovery
    Assumes CPMG condition, i.e. all states are real-valued.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians
        T1, T2 = Relaxation times (same as TE)
        TE = Echo Time interval (same as T1, T2)
        B1 = fraction of B1 homogeneity (1 is fully homogeneous)
        noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!
        recovery = True to include T1 recovery in the Z0 state.

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.

   """

    FpFmZ, EE = relax2(FpFmZ, TE/2., T1, T2)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf_B1_prime(FpFmZ, alpha, B1)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = torch.mm(EE, FpFmZ)

    return FpFmZ



### Gradients of full FSE EPG function across T time points


def FSE_signal_prime_alpha_idx(angles_rad, TE, T1, T2, idx):
    """Gradient of EPG function at each time point w.r.t. RF pulse alpha_i"""

    T = len(angles_rad)
    zi = torch.cat((np.array([[1],[1],[0]]), np.zeros((3, T))),1)

    z_prime = torch.zeros((T, 1))

    for i in range(T):
        alpha = angles_rad[i]
        if i < idx:
            zi = FSE_TE(zi, alpha, TE, T1, T2, noadd=True)
            z_prime[i] = 0
        elif i == idx:
            wi = FSE_TE_prime_alpha(zi, alpha, TE, T1, T2, noadd=True)
            z_prime[i] = wi[0,0]
        else:
            wi = FSE_TE(wi, alpha, TE, T1, T2, noadd=True, recovery=False)
            z_prime[i] = wi[0,0]

    return z_prime


def FSE_signal_prime_T1(angles_rad, TE, T1, T2):
    return FSE_signal_ex_prime_T1(np.pi/2, angles_rad, TE, T1, T2)

def FSE_signal_ex_prime_T1(angle_ex_rad, angles_rad, TE, T1, T2, B1=1.):
    """Gradient of EPG function at each time point w.r.t. T1"""
    
    T = len(angles_rad)

    try:
        B1 = B1[0]
    except:
        pass

    # since the grad doesn't depend on B1 inhomog, can just pre-scale flip angles
    angle_ex_rad = B1 * angle_ex_rad
    angles_rad = B1 * angles_rad
    
    zi = torch.cat((rf_ex(np.array([[0],[0],[1]]), angle_ex_rad), np.zeros((3, T))),1)
    z_prime = torch.zeros((T, 1))
    
    for i in range(T):
        
        alpha = angles_rad[i]

        if i == 0:
            wi = torch.zeros((3, T+1))
        else:
            wi = FSE_TE(wi, alpha, TE, T1, T2, noadd=True, recovery=False)
            
        wi += FSE_TE_prime1_T1(zi, alpha, TE, T1, T2, noadd=True)
        wi += FSE_TE_prime2_T1(zi, alpha, TE, T1, T2, noadd=True)

        zi = FSE_TE(zi, alpha, TE, T1, T2, noadd=True)
        z_prime[i] = wi[0,0]

    return z_prime


def FSE_signal_prime_T2(angles_rad, TE, T1, T2):
    return FSE_signal_ex_prime_T2(np.pi/2, angles_rad, TE, T1, T2)

def FSE_signal_ex_prime_T2(angle_ex_rad, angles_rad, TE, T1, T2, B1=1.):
    """Gradient of EPG function at each time point w.r.t. T2"""
    
    T = len(angles_rad)

    try:
        B1 = B1[0]
    except:
        pass

    # since the grad doesn't depend on B1 inhomog, can just pre-scale flip angles
    angle_ex_rad = B1 * angle_ex_rad
    angles_rad = B1 * angles_rad
    
    zi = torch.cat((rf_ex(np.array([[0],[0],[1]]), angle_ex_rad), np.zeros((3, T))),1)
    z_prime = torch.zeros((T, 1))

    for i in range(T):
      
        alpha = angles_rad[i]

        if i == 0:
            wi = torch.zeros((3, T+1))
        else:
            wi = FSE_TE(wi, alpha, TE, T1, T2, noadd=True, recovery=False)

        wi += FSE_TE_prime1_T2(zi, alpha, TE, T1, T2, noadd=True)
        wi += FSE_TE_prime2_T2(zi, alpha, TE, T1, T2, noadd=True)
        
        zi = FSE_TE(zi, alpha, TE, T1, T2, noadd=True)
        z_prime[i] = wi[0,0]

    return z_prime


def FSE_signal_ex_prime_B1(angle_ex_rad, angles_rad, TE, T1, T2, B1):
    """Gradient of EPG function at each time point w.r.t. B1 Homogeneity.
    Includes the excitation flip angle"""
    
    T = len(angles_rad)
    zi = torch.cat((np.array([[0],[0],[1]]), np.zeros((3, T+1))),1)

    z_prime = torch.zeros((T, 1))

    wi = rf_ex_B1_prime(zi, angle_ex_rad, B1)
    zi = rf_ex(zi, angle_ex_rad * B1)

    for i in range(T):

        alpha = angles_rad[i]

        if i == 0:
            xi = FSE_TE(wi, alpha * B1, TE, T1, T2, noadd=True, recovery=False)
        else:
            xi = FSE_TE(wi, alpha * B1, TE, T1, T2, noadd=True)

        wi = FSE_TE_prime_B1(zi, alpha, TE, T1, T2, B1, noadd=True) + xi

        zi = FSE_TE(zi, alpha * B1, TE, T1, T2, noadd=True)

        z_prime[i] = wi[0,0]

    return z_prime



### Full FSE EPG function across T time points


def FSE_signal_ex(angle_ex_rad, angles_rad, TE, T1, T2, B1=1.):
    """Same as FSE_signal2_ex, but only returns Mxy"""
    return FSE_signal2_ex(angle_ex_rad, angles_rad, TE, T1, T2, B1)[0]

def FSE_signal(angles_rad, TE, T1, T2):
    """Same as FSE_signal2, but only returns Mxy"""

    return FSE_signal2(angles_rad, TE, T1, T2)[0]

def FSE_signal2(angles_rad, TE, T1, T2):
    """Same as FSE_signal2_ex, but assumes excitation pulse is 90 degrees"""

    return FSE_signal2_ex(pi/2., angles_rad, TE, T1, T2)


def FSE_signal2_ex(angle_ex_rad, angles_rad, TE, T1, T2, B1=1.):
    """Simulate Fast Spin-Echo CPMG sequence with specific flip angle train.
    Prior to the flip angle train, an excitation pulse of angle_ex_rad degrees
    is applied in the Y direction. The flip angle train is then applied in the X direction.

    INPUT:
        angles_rad = array of flip angles in radians equal to echo train length
        TE = echo time/spacing
        T1 = T1 value in seconds
        T2 = T2 value in seconds

    OUTPUT:
        Mxy = Transverse magnetization at each echo time
        Mz = Longitudinal magnetization at each echo time
        
    """

    T = len(angles_rad)
#     Mxy = torch.zeros((T,1)).cuda()
#     Mz = torch.zeros((T,1)).cuda()
    n_pixel = T1.shape[0]
    Mxy = torch.zeros((T,n_pixel)).cuda()
    Mz = torch.zeros((T,n_pixel)).cuda()
    P = torch.Tensor([[0.],[0.],[1.]]).repeat(1,n_pixel).cuda() # initially on Mz
    
#     P = torch.Tensor([[0.],[0.],[1.]]).cuda() # initially on Mz

    try:
        B1 = B1[0]
    except:
        pass

    # pre-scale by B1 homogeneity
    angle_ex_rad = B1 * angle_ex_rad
    angles_rad = B1 * angles_rad

    P = rf_ex(P, angle_ex_rad) # initial tip

    for i in range(T):
        alpha = angles_rad[i]
        P = FSE_TE(P, alpha, TE, T1, T2)

        Mxy[i] = P[0,:]
        Mz[i] = P[2,:]

    return Mxy, Mz



def SE_sim(angle_ex_rad, angles_rad, TE, T1, T2, TR, B1=1.):
    Mxy, Mz = FSE_signal2_ex(angle_ex_rad, angles_rad, TE, T1, T2, B1=1.)
    par = 1 - torch.exp(-(TR - TE)/T1)
#     print(type(Mxy))
#     return Mxy * par.float().cuda(), Mz    
    return Mxy * par.float(), Mz






