#!/usr/bin/python

# T1-T2 Shuffling signal equation model, including gradients w.r.t. primary tissue parameters
# 2016-2017 Jon Tamir <jtamir@eecs.berkeley.edu>

from __future__ import division
import numpy as np
from numpy import pi, exp
import epg.epgcpmg as epg

def T1_recovery(T, T1):
    return exp(-T/T1)

def T1_recovery_prime(T, T1):
    return T * exp(-T/T1) / T1**2

def t1t2shuffle(angles_rad, TE, TRs, M0, T1, T2):
    """T1-T2 Shuffling signal equation: 3DFSE with driven equillibrium"""

    return t1t2shuffle_ex(pi/2., angles_rad, TE, TRs, M0, T1, T2)

def t1t2shuffle_ex(angle_ex_rad, angles_rad, TE, TRs, M0, T1, T2, B1=1.):
    """Same as t1t2shuffle, with arbitrary RF excitation flip angle"""

    T = angles_rad.size

    URs = TRs - (T+1)*TE # T + 1 to account for fast recovery

    fi = epg.FSE_signal_ex(angle_ex_rad, angles_rad, TE, T1, T2, B1)
    Ej = T1_recovery(URs, T1)[None,:]

    sig = M0 * fi * (1 - Ej) / (1 - fi[-1]*Ej)
    
    return sig.ravel(order='F')

    
def t1t2shuffle_prime_T2(angles_rad, TE, TRs, M0, T1, T2, B1=1.):
    """derivative of signal equation w.r.t. T2"""
    return t1t2shuffle_ex_prime_T2(pi/2, angles_rad, TE, TRs, M0, T1, T2, B1)

def t1t2shuffle_ex_prime_T2(angle_ex_rad, angles_rad, TE, TRs, M0, T1, T2, B1=1.):
    """derivative of signal equation w.r.t. T2, with arbitrary RF excitation"""

    T = angles_rad.size

    URs = TRs - (T+1)*TE # T + 1 to account for fast recovery

    fi = epg.FSE_signal_ex(angle_ex_rad, angles_rad, TE, T1, T2, B1)
    fi_prime = epg.FSE_signal_ex_prime_T2(angle_ex_rad, angles_rad, TE, T1, T2, B1)
    
    Ej = T1_recovery(URs, T1)[None,:]
    
    sig_prime = M0 * (1 - Ej) * (fi_prime * (1 - Ej*fi[-1]) + Ej*fi_prime[-1]*fi) / (1 - Ej*fi[-1])**2

    return sig_prime.ravel(order='F')

def t1t2shuffle_prime_T1(angles_rad, TE, TRs, M0, T1, T2, B1=1.):
    """derivative of signal equation w.r.t. T1"""
    return t1t2shuffle_ex_prime_T1(pi/2, angles_rad, TE, TRs, M0, T1, T2, B1)

def t1t2shuffle_ex_prime_T1(angle_ex_rad, angles_rad, TE, TRs, M0, T1, T2, B1=1.):
    """derivative of signal equation w.r.t. T1, with arbitrary RF excitation"""
    
    T = angles_rad.size

    URs = TRs - (T+1)*TE # T + 1 to account for fast recovery
    
    fi = epg.FSE_signal_ex(angle_ex_rad, angles_rad, TE, T1, T2, B1)
    fi_prime = epg.FSE_signal_ex_prime_T1(angle_ex_rad, angles_rad, TE, T1, T2, B1)
    
    Ej = T1_recovery(URs, T1)[None,:] 
    Ej_prime = T1_recovery_prime(URs, T1)[None, :]
    
    sig_prime = M0 * ((fi_prime*(1 - Ej) - Ej_prime*fi) * (1 - Ej*fi[-1]) + (Ej_prime*fi[-1] + Ej*fi_prime[-1])*fi*(1-Ej)) / (1 - Ej*fi[-1])**2
    
    return sig_prime.ravel(order='F')

def t1t2shuffle_prime_M0(angles_rad, TE, TRs, M0, T1, T2, B1=1.):
    """derivative of signal equation w.r.t. M0"""
    return t1t2shuffle_ex_prime_M0(pi/2, angles_rad, TE, TRs, 1., T1, T2, B1)

def t1t2shuffle_ex_prime_M0(angle_ex_rad, angles_rad, TE, TRs, M0, T1, T2, B1=1.):
    """derivative of signal equation w.r.t. M0, with arbitrary RF excitation"""
    return t1t2shuffle_ex(angle_ex_rad, angles_rad, TE, TRs, 1., T1, T2, B1)

def t1t2shuffle_ex_prime_B1(angle_ex_rad, angles_rad, TE, TRs, M0, T1, T2, B1=1.):
    """derivative of signal equation w.r.t. B1, with arbitrary RF excitation"""
    
    T = angles_rad.size

    URs = TRs - (T+1)*TE # T + 1 to account for fast recovery

    fi = epg.FSE_signal_ex(angle_ex_rad, angles_rad, TE, T1, T2, B1)
    fi_prime = epg.FSE_signal_ex_prime_B1(angle_ex_rad, angles_rad, TE, T1, T2, B1)

    Ej = T1_recovery(URs, T1)[None,:] 

    a = 1 - Ej
    b = 1 - Ej * fi[-1]

    sig_prime = M0 * a / b**2 * (fi_prime * b + fi * Ej * fi_prime[-1])
    
    return sig_prime.ravel(order='F')

def t1t2shuffle_prime_alpha_idx(angles_rad, TE, TRs, M0, T1, T2, idx):
    """derivative of signal equation w.r.t. the i'th flip angle"""

    T = angles_rad.size

    URs = TRs - (T+1)*TE # T + 1 to account for fast recovery

    fi = epg.FSE_signal(angles_rad, TE, T1, T2)
    fi_prime = epg.FSE_signal_prime_alpha_idx(angles_rad, TE, T1, T2, idx)

    Ej = T1_recovery(URs, T1)[None,:] 

    a = 1 - Ej
    b = 1 - Ej * fi[-1]

    sig_prime = M0 * a / b**2 * (fi_prime * b + fi * Ej * fi_prime[-1])
    
    return sig_prime.ravel(order='F')
