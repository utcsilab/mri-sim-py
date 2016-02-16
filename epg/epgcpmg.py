#!/usr/bin/python

# EPG CPMG simulation code, based off of Matlab scripts from Brian Hargreaves <bah@stanford.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>

from __future__ import division
import numpy as np
from numpy import pi, cos, sin, exp, conj
from warnings import warn

def rf(FpFmZ, alpha):
    "Same as rf2, but only returns FpFmZ"""
    return rf2(FpFmZ, alpha)[0]

def rf2(FpFmZ, alpha):
    """ Propagate EPG states through an RF rotation of 
    alpha (radians). Assumes CPMG condition, i.e.
    magnetization lies on the real x axis.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians

    OUTPUT:
        FpFmZ = Updated FpFmZ state.
        RR = RF rotation matrix (3x3).

    """

    # -- From Weigel at al, JMRI 41(2015)266-295, Eq. 21.
    
    if abs(alpha) > 2 * pi:
        warn('rf2: Flip angle should be in radians!')

    cosa2 = cos(alpha/2.)**2
    sina2 = sin(alpha/2.)**2

    cosa = cos(alpha)
    sina = sin(alpha)

    RR = np.array([ [cosa2, sina2, sina],
                    [sina2, cosa2, -sina],
                    [-0.5 * sina, 0.5 * sina, cosa] ])


    FpFmZ = np.dot(RR, FpFmZ)

    return FpFmZ, RR



def rf_prime(FpFmZ, alpha):
    """Same as rf_prime, but only returns FpFmZ"""
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

    if abs(alpha) > 2 * pi:
        warn('rf2: Flip angle should be in radians!')

    RR = np.array([ [-cos(alpha/2.) * sin(alpha/2.), cos(alpha/2.) * sin(alpha/2.), cos(alpha)],
                    [cos(alpha/2.) * sin(alpha/2.), -cos(alpha/2.) * sin(alpha/2.), -cos(alpha)],
                    [-0.5 * cos(alpha), 0.5 * cos(alpha), -sin(alpha)] ])

    FpFmZ = np.dot(RR, FpFmZ)

    return FpFmZ, RR



def relax_mat(T, T1, T2):
    E2 = exp(-T/T2)
    E1 = exp(-T/T1)

    EE = np.diag([E2, E2, E1])      # Decay of states due to relaxation alone.

    return EE


def relax(FpFmZ, T, T1, T2):
    """Same as relax2, but only returns FpFmZ"""
    return relax2(FpFmZ, T, T1, T2)[0]

def relax2(FpFmZ, T, T1, T2):
    """ Propagate EPG states through a period of relaxation over
    an interval T.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        T1, T2 = Relaxation times (same as T)
        T = Time interval (same as T1,T2)

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.
        EE = decay matrix, 3x3 = diag([E2 E2 E1]);

   """

    E2 = exp(-T/T2)
    E1 = exp(-T/T1)

    EE = np.diag([E2, E2, E1])      # Decay of states due to relaxation alone.
    RR = 1 - E1                     # Mz Recovery, affects only Z0 state, as 
                                    # recovered magnetization is not dephased.


    FpFmZ = np.dot(EE, FpFmZ)       # Apply Relaxation
    FpFmZ[2,0] = FpFmZ[2,0] + RR    # Recovery  

    return FpFmZ, EE



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
        FpFmZ = np.hstack((FpFmZ, [[0],[0],[0]]))   # add higher dephased state

    FpFmZ[0,:] = np.roll(FpFmZ[0,:], 1)     # shift Fp states
    FpFmZ[1,:] = np.roll(FpFmZ[1,:], -1)    # shift Fm states
    FpFmZ[1,-1] = 0                         # Zero highest Fm state
    FpFmZ[0,0] = FpFmZ[1,0]                 # Fill in lowest Fp state

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

    EE = relax_mat(TE/2., T1, T2)

    if recovery:
        FpFmZ = relax(FpFmZ, TE/2., T1, T2)
    else:
        FpFmZ = np.dot(EE, FpFmZ)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf(FpFmZ, alpha)
    FpFmZ = grad(FpFmZ, noadd)
    if recovery:
        FpFmZ = relax(FpFmZ, TE/2., T1, T2)
    else:
        FpFmZ = np.dot(EE, FpFmZ)

    return FpFmZ


def FSE_TE_prime(FpFmZ, alpha, TE, T1, T2, noadd=False, recovery=True):
    """ Gradient of EPG propagatopm pver a full TE, i.e.
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
    FpFmZ = np.dot(EE, FpFmZ)

    return FpFmZ



def FSE_signal(angles_rad, TE, T1, T2):
    """Same as FSE_signal2, but only returns Mxy"""
    return FSE_signal2(angles_rad, TE, T1, T2)[0]

def FSE_signal2(angles_rad, TE, T1, T2):
    """Simulate Fast Spin-Echo CPMG sequence with specific flip angle train.
    Prior to the flip angle train, a 90 degree pulse is applied in the Y direction.
    The flip angle train is then applied in the X direction.

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
    Mxy = np.zeros((T,1))
    Mz = np.zeros((T,1))

    P = np.array([[1],[1],[0]])  # initial tip


    for i in range(T):
        alpha = angles_rad[i]
        P = FSE_TE(P, alpha, TE, T1, T2)

        Mxy[i] = P[0,0]
        Mz[i] = P[2,0]

    return Mxy, Mz



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T1 = 1000e-3
    T2 = 200e-3

    TE = 5e-3

    N = 100
    angles = 120 * np.ones((N,))
    angles_rad = angles * pi / 180.

    S = FSE_signal(angles_rad, TE, T1, T2)
    S2 = abs(S)
    plt.plot(TE*1000*np.arange(1, N+1), S2)
    plt.xlabel('time (ms)')
    plt.ylabel('signal')
    plt.title('T1 = %.2f ms, T2 = %.2f ms' % (T1 * 1000, T2 * 1000))
    plt.show()
    

