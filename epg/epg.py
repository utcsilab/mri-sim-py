#!/usr/bin/python

# EPG Simulation code, based off of Matlab scripts from Brian Hargreaves <bah@stanford.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>

import numpy as np
from numpy import pi, cos, sin, exp, conj
from warnings import warn

def epg_rf(FpFmZ, alpha, phi):
    """ Propagate EPG states through an RF rotation of 
    alpha, with phase phi (both radians).

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.

    OUTPUT:
        FpFmZ = Updated FpFmZ state.
        RR = RF rotation matrix (3x3).

    SEE ALSO:
    epg_grad, epg_grelax
    """

    # -- From Weigel at al, JMR 205(2010)276-285, Eq. 8.
    
    if abs(alpha) > 2 * pi:
        warn('epg_rf: Flip angle should be in radians!')

    RR = np.array([ [ (cos(alpha/2.))**2, exp(2*1j*phi)*(sin(alpha/2.))**2, -1j*exp(1j*phi)*sin(alpha)],
            [exp(-2*1j*phi)*(sin(alpha/2.))**2, (cos(alpha/2.))**2, 1j*exp(-1j*phi)*sin(alpha)],
            [-1j/2.*exp(-1j*phi)*sin(alpha), 1j/2.*exp(1j*phi)*sin(alpha), cos(alpha)] ])

    FpFmZ = np.dot(RR , FpFmZ)

    return FpFmZ, RR


def epg_relax(FpFmZ, T1, T2, T):
    """ Propagate EPG states through a period of relaxation over
    an interval T.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        T1,T2 = Relaxation times (same as T)
        T = Time interval (same as T1,T2)

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.
        EE = decay matrix, 3x3 = diag([E2 E2 E1]);

    SEE ALSO:
           epg_grad, epg_rf
   """

    E2 = exp(-T/T2)
    E1 = exp(-T/T1)

    EE = np.diag([E2, E2, E1])      # Decay of states due to relaxation alone.
    RR = 1 - E1                     # Mz Recovery, affects only Z0 state, as 
                                    # recovered magnetization is not dephased.


    FpFmZ = np.dot(EE, FpFmZ)       # Apply Relaxation
    FpFmZ[2,0] = FpFmZ[2,0] + RR    # Recovery  

    return FpFmZ, EE


def epg_grad(FpFmZ, noadd=False):
    """Propagate EPG states through a "unit" gradient.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!

    OUTPUT:
        Updated FpFmZ state.

    SEE ALSO:
        epg_grad, epg_grelax
    """

    # Gradient does not affect the Z states.

    if noadd == False:
        FpFmZ = np.hstack((FpFmZ, [[0],[0],[0]]))   # add higher dephased state

    FpFmZ[0,:] = np.roll(FpFmZ[0,:], 1)     # shift Fp states
    FpFmZ[1,:] = np.roll(FpFmZ[1,:], -1)    # shift Fm states
    FpFmZ[1,-1] = 0                         # Zero highest Fm state
    FpFmZ[0,0] = conj(FpFmZ[1,0])           # Fill in lowest Fp state

    return FpFmZ


def FSE_signal(angles_rad, TE, T1, T2):
    """Simulate Fast Spin Echo sequence with specific flip angle sequence.

    INPUT:
        angles_rad = array of flip angles in radians equal to echo train length
        TE = echo time/spacing
        T1 = T1 value in seconds
        T2 = T2 value in seconds

    OUTPUT:
        Complex signal value at each echo time
        
    """

    T = len(angles_rad)
    S = np.zeros((T,1), dtype=complex)

    P = np.array([[0],[0],[1]])    # initially in M0

    P = epg_rf(P, pi/2, pi/2)[0]    # 90 degree tip

    for i in range(T):
        alpha = angles_rad[i]
        P = epg_relax(P, T1, T2, TE/2.)[0]
        P = epg_grad(P)
        P = epg_rf(P, alpha, 0)[0]
        P = epg_relax(P, T1, T2, TE/2.)[0]
        P = epg_grad(P)

        S[i] = P[0,0]

    return S
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T1 = 1000e-3
    T2 = 200e-3

    TE = 5e-3

    N = 100
    angles = 180 * np.ones((N,))
    angles_rad = angles * pi / 180.

    S = FSE_signal(angles_rad, TE, T1, T2)
    S2 = abs(S)
    plt.plot(TE*1000*np.arange(1, N+1), S2)
    plt.xlabel('time (ms)')
    plt.ylabel('signal')
    plt.title('T1 = %.2f ms, T2 = %.2f ms' % (T1 * 1000, T2 * 1000))
    plt.show()
    

