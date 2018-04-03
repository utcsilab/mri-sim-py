#!/usr/bin/python

# EPG Simulation code, based off of Matlab scripts from Brian Hargreaves <bah@stanford.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>

from __future__ import division
import numpy as np
from numpy import pi, cos, sin, exp, conj
from warnings import warn

def rf(FpFmZ, alpha, phi):
    "Same as rf2, but only returns FpFmZ"""
    return rf2(FpFmZ, alpha, phi)[0]

def rf2(FpFmZ, alpha, phi):
    """ Propagate EPG states through an RF rotation of 
    alpha, with phase phi (both radians).

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha, phi = RF pulse flip angle phase in radians

    OUTPUT:
        FpFmZ = Updated FpFmZ state.
        RR = RF rotation matrix (3x3).

    """

    # -- From Weigel at al, JMRI 41(2015)266-295, Eq. 15.
    
    if abs(alpha) > 2 * pi:
        warn('rf: Flip angle should be in radians!')

    cosa2 = cos(alpha/2.)**2
    sina2 = sin(alpha/2.)**2
    cosa = cos(alpha)
    sina = sin(alpha)

    ejp = exp(1j * phi)
    e2jp = exp(2j * phi)
    emjp = exp(-1j * phi)
    em2jp = exp(-2j * phi)

    RR = np.array([ [ cosa2, e2jp * sina2, -1j * ejp * sina],
            [em2jp * sina2, cosa2, 1j * emjp * sina],
            [-1j/2. * emjp * sina, 1j/2. * ejp * sina, cosa] ])


    FpFmZ = np.dot(RR , FpFmZ)

    return FpFmZ, RR


def rf_prime(FpFmZ, alpha, phi):
    """Same as rf_prime, but only returns FpFmZ"""
    return rf_prime2(FpFmZ, alpha, phi)[0]

def rf_prime2(FpFmZ, alpha, phi):
    """ Compute the gradient of the RF rotation operator w.r.t. alpha, where
    alpha and phase (radians) are the RF rotation and phase. Phase is assumed
    to be a constant

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha, phi = RF pulse flip angle and phase in radians

    OUTPUT:
        FpFmZ = Derivative of FpFmZ state w.r.t. alpha
        RR = Derivative of RF rotation matrix (3x3) w.r.t. alpha

    """

    if abs(alpha) > 2 * pi:
        warn('rf2: Flip angle should be in radians!')


    cosa2 = cos(alpha/2.)
    sina2 = sin(alpha/2.)
    cosa = cos(alpha)
    sina = sin(alpha)

    ejp = exp(1j * phi)
    e2jp = exp(2j * phi)
    emjp = exp(-1j * phi)
    em2jp = exp(-2j * phi)


    RR = np.array([ [-cosa2 * sina2, e2jp * cosa2 * sina2, -1j * ejp * cosa],
                    [em2jp * cosa2 * sina2, -cosa2 * sina2, 1j * emjp * cosa],
                    [-0.5j * emjp * cosa, 0.5j * ejp * cosa, -sina] ])


    FpFmZ = np.dot(RR, FpFmZ)

    return FpFmZ, RR


def relax_mat(T, T1, T2):
    E2 = exp(-T/T2)
    E1 = exp(-T/T1)

    EE = np.diag([E2, E2, E1])      # Decay of states due to relaxation alone.

    return EE

def relax(FpFmZ, T, T1, T2):
    """ Propagate EPG states through a period of relaxation over
    an interval T.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        T1, T2 = Relaxation times (same as T)
        T = Time interval (same as T1,T2)

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.

   """
   
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
    """Propagate EPG states through a "unit" gradient.

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
    FpFmZ[0,0] = conj(FpFmZ[1,0])           # Fill in lowest Fp state

    return FpFmZ


def FSE_TE(FpFmZ, alpha, phi, TE, T1, T2, noadd=False, recovery=True):
    """ Propagate EPG states through a full TE, i.e.
    relax -> grad -> rf -> grad -> relax

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha, phi = RF pulse flip angle phase in radians
        T1, T2 = Relaxation times (same as T)
        TE = Echo Time interval (same as T1,T2)
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
    FpFmZ = rf(FpFmZ, alpha, phi)
    FpFmZ = grad(FpFmZ, noadd)
    if recovery:
        FpFmZ = relax(FpFmZ, TE/2., T1, T2)
    else:
        FpFmZ = np.dot(EE, FpFmZ)

    return FpFmZ


def FSE_TE_prime(FpFmZ, alpha, phi, TE, T1, T2, noadd=False, recovery=True):
    """ Gradient of EPG propagatiom over a full TE, i.e.
    relax -> grad -> rf_prime -> grad -> relax_hat,
    where rf_prime is the derivative of the RF pulse matrix w.r.t. alpha,
    and relax_hat  is the relaxation without longitudinal recovery

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha, phi = RF pulse flip angle and phase in radians
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
    FpFmZ = rf_prime(FpFmZ, alpha, phi)
    FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = np.dot(EE, FpFmZ)

    return FpFmZ



def FSE_signal(angles_rad, phase_rad, TE, T1, T2, M0=np.array([[0.],[0.],[1.]]), excitation_dict=None, inversion_dict=None):
    """Same as FSE_signal2, but only returns Mxy"""
    return FSE_signal2(angles_rad, phase_rad, TE, T1, T2, M0, excitation_dict=excitation_dict, inversion_dict=inversion_dict)[0]

def FSE_signal2(angles_rad, phase_rad, TE, T1, T2, M0=np.array([[0.],[0.],[1.]]), excitation_dict=None, inversion_dict=None):
    """Simulate Fast Spin-Echo sequence with specific flip angle/phase train.
    The first flip angle is applied at time t=0. The second flip angle is applied at
    time t=TE/2. Subsequent flip angles are spaced TE seconds apart.

    INPUT:
        angles_rad, phase_rad = array of flip angles and phase in radians
        TE = echo time/spacing
        T1 = T1 value in seconds
        T2 = T2 value in seconds

    OUTPUT:
        Mxy = Transverse magnetization at each echo time
        Mz = Longitudinal magnetization at each echo time
        
    """

    T = len(angles_rad)
    Mxy = np.zeros((T,1), dtype=complex)
    Mz = np.zeros((T,1))

    P = np.copy(M0)

    if inversion_dict != None:
        alpha_TI, phi_TI, TI = inversion_dict['alpha_rad'], inversion_dict['phi_rad'], inversion_dict['TI']
        P = rf(P, alpha_TI, phi_TI) # apply inversion pulse
        P = relax(P, TI, T1, T2) # wait inversion time

    if excitation_dict != None:
        alpha_e, phi_e = excitation_dict['alpha_rad'], excitation_dict['phi_rad']
        P = rf(P, alpha_e, phi_e) # apply excitation pulse
    else:
        P = np.array([[1.], [1.], [0.]]) # 90 degree CPMG excitation

    Mxy[0] = P[0,0]
    Mz[0] = np.real(P[2,0])

    for i in xrange(T):

        alpha = angles_rad[i]
        phi = phase_rad[i]
        P = FSE_TE(P, alpha, phi, TE, T1, T2)

        Mxy[i] = P[0,0]
        Mz[i] = np.real(P[2,0]) # real up to machine precision

    return Mxy, Mz

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T1 = 1000e-3
    T2 = 200e-3

    TE = 5e-3

    N = 100
    angles = 120 * np.ones((N,))
    angles_rad = angles * pi / 180.

    S = FSE_CPMG(angles_rad, TE, T1, T2)
    S2 = abs(S)
    plt.plot(TE*1000*np.arange(1, N+1), S2)
    plt.xlabel('time (ms)')
    plt.ylabel('signal')
    plt.title('T1 = %.2f ms, T2 = %.2f ms' % (T1 * 1000, T2 * 1000))
    plt.show()
    

