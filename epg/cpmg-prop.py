#!/usr/bin/python

import numpy as np
from numpy import pi, cos, sin, exp, conj
from warnings import warn
import epgcpmg as epg
import sys


class PulseTrain:
    def __init__(self, T, TE, loss_fun, loss_fun_prime, angles_rad=None):
        self.T = T
        self.TE = TE
        self.loss_fun = loss_fun
        self.loss_fun_prime = loss_fun_prime
        if angles_rad is None:
            self.angles_rad = angles_rad
        else:
            self.reset()

    def reset(self):
        self.angles_rad = pi / 180 * (50 + (120 - 50) * np.random.rand(self.T))

    def forward(self, theta):
        T = self.T
        P = np.zeros((self.T, 3, 2 * T + 1))

        P0= np.hstack((np.array([[1],[1],[0.]]), np.zeros((3, 2 * T)))) # initial tip

        for i in range(T):
            alpha = self.angles_rad[i]
            T1 = theta['T1']
            T2 = theta['T2']
            TE = self.TE

            if i == 0:
                P[0, :, :] = epg.FSE_TE(P0, alpha, T1, T2, TE, noadd=True)
            else:
                P[i, :, :] = epg.FSE_TE(P[i - 1, :, :], alpha, T1, T2, TE, noadd=True)

        return P


def loss(theta1, theta2, angles_rad):
    x1 = epg.FSE_signal(angles_rad, TE, theta1['T1'], theta1['T2'])
    x2 = epg.FSE_signal(angles_rad, TE, theta2['T1'], theta2['T2'])

    return 0.5 * np.linalg.norm(x1, ord=2)**2 + 0.5 * np.linalg.norm(x2, ord=2)**2 - np.dot(x1.ravel(), x2.ravel())


def loss_prime(theta1, theta2, angles_rad):
    x1 = epg.FSE_signal(angles_rad, TE, theta1['T1'], theta1['T2']).ravel()
    x2 = epg.FSE_signal(angles_rad, TE, theta2['T1'], theta2['T2']).ravel()

    T = len(angles_rad)
    alpha_prime = np.zeros((T, 1))

    for i in range(T):
        x1_prime = sig_prime_i(theta1, angles_rad, i).ravel()
        x2_prime = sig_prime_i(theta2, angles_rad, i).ravel()
        M1 = np.dot(x1, x1_prime)
        M2 = np.dot(x2, x2_prime)
        M3 = np.dot(x1, x2_prime)
        M4 = np.dot(x2, x1_prime)

        alpha_prime[i] = M1 + M2 - M3 - M4

    return alpha_prime

#
    #M1 = x1 * x1_prime
    #M2 = x2 * x2_prime
    #M3 = x1 * x2_prime
    #M4 = x2 * x1_prime
#
    #return M1 + M2 - M3 - M4


def sig_prime_i(theta, angles_rad, idx):
    T1, T2 = get_params(theta)
    T = len(angles_rad)
    zi = np.hstack((np.array([[1],[1],[0]]), np.zeros((3, T))))

    z_prime = np.zeros((T, 1))

    for i in range(T):
        alpha = angles_rad[i]
        print 'z[%d] = FSE_TE(z[%d]' % (i, i-1)
        if i < idx:
            zi = epg.FSE_TE(zi, alpha, TE, T1, T2, noadd=True)
            z_prime[i] = 0
            print 'z_prime[%d] = 0' % i
        elif i == idx:
            wi = epg.FSE_TE_prime(zi, alpha, TE, T1, T2, noadd=True)
            print 'w[%d] = FSE_TE_PRIME(z[%d]' % (i, i-1)
            z_prime[i] = wi[0,0]
            print 'z_prime[%d] = w[%d]' % (i, i)
        else:
            wi = epg.FSE_TE(wi, alpha, TE, T1, T2, noadd=True, recovery=False)
            print 'w[%d] = FSE_TE(w[%d]' % (i, i-1)
            print 'z_prime[%d] = w[%d]' % (i, i)
            z_prime[i] = wi[0,0]
    print

    return z_prime



    #for i in range(T):
        #alpha = angles_rad[i]
        #print 'w[%d] = FSE_TE_PRIME(z[%d]' % (i, i-1)
        #print 'z[%d] = FSE_TE(z[%d]' % (i, i-1)
        #wi = epg.FSE_TE_prime(zi, alpha, TE, T1, T2, noadd=True)
        #zi = epg.FSE_TE(zi, alpha, TE, T1, T2, noadd=True)
        #beta[i, :, :] = wi
        #for j in range(i):
            #print 'b[%d] = FSE_TE(b[%d]' % (j, j)
            #beta[j, :, :] = epg.FSE_TE(beta[j, :, :], alpha, TE, T1, T2, noadd=True, recovery=False)

    #for i in range(T):
        #angles_prime[i] = beta[i,0,0]

    #return angles_prime


def sig_prime(theta, angles_rad):
    T1, T2 = get_params(theta)
    T = len(angles_rad)
    zi = np.hstack((np.array([[1],[1],[0]]), np.zeros((3, T))))

    angles_prime = np.zeros((T, 1))
    beta = np.zeros((T, 3, T+1))

    for i in range(T):
        alpha = angles_rad[i]
        print 'w[%d] = FSE_TE_PRIME(z[%d]' % (i, i-1)
        print 'z[%d] = FSE_TE(z[%d]' % (i, i-1)
        wi = epg.FSE_TE_prime(zi, alpha, TE, T1, T2, noadd=True)
        zi = epg.FSE_TE(zi, alpha, TE, T1, T2, noadd=True)
        beta[i, :, :] = wi
        for j in range(i):
            print 'b[%d] = FSE_TE(b[%d]' % (j, j)
            beta[j, :, :] = epg.FSE_TE(beta[j, :, :], alpha, TE, T1, T2, noadd=True, recovery=False)

    for i in range(T):
        angles_prime[i] = beta[i,0,0]

    return angles_prime

def get_params(theta):
    return theta['T1'], theta['T2']


def numerical_gradient(theta1, theta2, angles_rad):
    initial_params = angles_rad
    num_grad = np.zeros(initial_params.shape)
    perturb = np.zeros(initial_params.shape)
    e = 1e-5

    for p in range(len(initial_params)):
        perturb[p] = e
        loss2 = loss(theta1, theta2, angles_rad + perturb)
        loss1 = loss(theta1, theta2, angles_rad - perturb)

        num_grad[p] = (loss2 - loss1) / (2 * e)

        perturb[p] = 0

    return num_grad


if __name__ == "__main__":
    #import matplotlib.pyplot as plt

    T1 = 1000e-3
    T2 = 200e-3

    TE = 5e-3

    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 10

    angles = 150 * np.ones((N,))
    angles_rad = angles * pi / 180.

    S = epg.FSE_signal(angles_rad, TE, T1, T2)
    S2 = abs(S)

    theta1 = {'T1': 1000e-3, 'T2': 200e-3}
    theta2 = {'T1': 1000e-3, 'T2': 500e-3}
    NG = numerical_gradient(theta1, theta2, angles_rad[:N])
    LP = loss_prime(theta1, theta2, angles_rad[:N])
    print 'Numerical Gradient:\t', NG
    print 'Analytical Gradient:\t', LP.T
    #plt.plot(TE*1000*np.arange(1, N+1), S2)
    #plt.xlabel('time (ms)')
    #plt.ylabel('signal')
    #plt.title('T1 = %.2f ms, T2 = %.2f ms' % (T1 * 1000, T2 * 1000))
    #plt.show()
    

