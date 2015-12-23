#!/usr/bin/python

import numpy as np
from numpy import pi, cos, sin, exp, conj
from warnings import warn
import epgcpmg as epg


class PulseTrain:
    def __init__(self, T, TE, loss_fun, loss_fun_prime, angles_rad=None):
        self.T = T
        self.TE = TE
        self.loss_fun = loss_fun
        self.loss_fun_prime = loss_fun_prime
        if angles_rad != None:
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
    x1 = epg.FSE_signal(angles_rad, TE, theta1['T1'], theta1['T2']).ravel()
    x2 = epg.FSE_signal(angles_rad, TE, theta2['T1'], theta2['T2']).ravel()

    return 0.5 * np.linalg.norm(x1, ord=2)**2 + 0.5 * np.linalg.norm(x2, ord=2)**2 - np.dot(x1, x2)


def loss_prime(theta1, theta2, angles_rad):
    x1 = epg.FSE_signal(angles_rad, TE, theta1['T1'], theta1['T2']).ravel()
    x2 = epg.FSE_signal(angles_rad, TE, theta2['T1'], theta2['T2']).ravel()

    M1 = x1 * sig_prime(theta1, angles_rad)
    M2 = x2 * sig_prime(theta2, angles_rad)
    M3 = x1 * sig_prime(theta2, angles_rad) + sig_prime(theta1, angles_rad) * x2
    return M1 + M2 - M3


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
    import matplotlib.pyplot as plt

    T1 = 1000e-3
    T2 = 200e-3

    TE = 5e-3

    N = 100
    angles = 120 * np.ones((N,))
    angles_rad = angles * pi / 180.

    S = epg.FSE_signal(angles_rad, TE, T1, T2)
    S2 = abs(S)
    plt.plot(TE*1000*np.arange(1, N+1), S2)
    plt.xlabel('time (ms)')
    plt.ylabel('signal')
    plt.title('T1 = %.2f ms, T2 = %.2f ms' % (T1 * 1000, T2 * 1000))
    plt.show()
    

