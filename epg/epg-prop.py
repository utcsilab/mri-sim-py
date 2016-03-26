#!/usr/bin/python

import numpy as np
from numpy import pi, cos, sin, exp, conj
from warnings import warn
import epg
import time
import sys
import scipy.io


class PulseTrain:
    def __init__(self, state_file, T, TE, TR, loss_fun, loss_fun_prime, angles_rad=None, phase_rad=None, verbose=False, step=.01, max_iter=100):
        self.state_file = state_file
        self.T = T
        self.TE = TE
        self.TR = TR
        self.loss_fun = loss_fun
        self.loss_fun_prime = loss_fun_prime
        self.max_iter = max_iter
        self.step = step
        self.verbose = verbose
        self.excitation_dict = None
        self.inversion_dict = None

        self.reset()
        if angles_rad is not None:
            self.set_angles_rad(angles_rad, phase_rad)

    def set_angles_rad(self, angles_rad, pase_rad):
        T = len(angles_rad)
        if T < self.T:
            self.angles_rad = np.hstack((angles_rad, np.zeros((self.T-T))))
            self.phase_rad = np.hstack((phase_rad, np.zeros((self.T-T))))
            self.phase_rad[:2] = np.pi/2
        else:
            self.angles_rad = angles_rad[:self.T]
            self.phase_rad = phase_rad[:self.T]

    def reset(self):
        self.angles_rad = DEG2RAD(50 + (120 - 50) * np.random.rand(self.T))
        self.phase_rad = np.zeros((self.T,))
        self.loss = []

    def save_state(self, filename=None):
        state = {
                'angles_rad': self.angles_rad,
                'phase_rad': self.phase_rad,
                'loss': self.loss,
                'max_iter': self.max_iter,
                'step': self.step,
                'T': self.T,
                'TE': self.TE,
                'verbose': self.verbose,
                }
        if filename is None:
            scipy.io.savemat(self.state_file, state, appendmat=False)
        else:
            scipy.io.savemat(filename, state, appendmat=False)

    def load_state(self, filename=None):
        if filename is None:
            state = scipy.io.loadmat(self.state_file)
        else:
            state = scipy.io.loadmat(filename)

        self.angles_rad = state['angles_rad'].ravel()
        self.phase_rad = state['phase_rad'].ravel()
        self.loss = list(state['loss'].ravel())
        self.max_iter = state['max_iter'].ravel()[0]
        self.step = state['step'].ravel()[0]
        self.T = state['T'].ravel()[0]
        self.TE = state['TE'].ravel()[0]
        self.verbose = state['verbose'].ravel()[0]


    def train(self, theta1, theta2):
        for i in range(self.max_iter):
            #angles_prime, angle_e_prime, angle_TI_prime = self.loss_fun_prime(theta1, theta2, self.angles_rad, self.phase_rad, self.TE, self.TR, self.excitation_dict, self.inversion_dict)
            angles_prime = self.loss_fun_prime(theta1, theta2, self.angles_rad, self.phase_rad, self.TE, self.TR)
            self.angles_rad = self.angles_rad + self.step * angles_prime
            #if self.excitation_dict != None:
                #self.anlge_e = self.angle_e + self.step * angle_e_prime
            #self.angle_TI = self.angle_TI + self.step * angle_TI_prime

            self.loss.append(self.loss_fun(theta1, theta2, self.angles_rad, self.phase_rad, self.TE, self.TR))
            str = '%d\t%3.3f' % (i, self.loss[-1])
            self.print_verbose(str)

    def print_verbose(self, str):
        if self.verbose:
            print str, RAD2DEG(self.angles_rad)

    def plot_vals(self, thetas):
        plt.subplot(2,1,1)
        plt.plot(range(self.T), RAD2DEG(self.angles_rad), 'o-')
        plt.xlim((0,self.T))
        plt.subplot(2,1,2)
        for theta in thetas:
            plt.plot(range(self.T), epg.FSE_signal(self.angles_rad, self.phase_rad, self.TE, theta['T1'], theta['T2']))
        plt.xlim((0,self.T))
        plt.ylim((0,1))

    def forward(self, theta):
        return epg.FSE_signal(self.angles_rad, self.phase_rad, TE, theta['T1'], theta['T2']).ravel()


def loss(theta1, theta2, angles_rad, phase_rad, TE, TR):
    T = len(angles_rad)
    
    x1 = epg.FSE_signal(angles_rad, phase_rad, TE, theta1['T1'], theta1['T2']) * (1 - exp(-(TR - T * TE)/theta1['T1']))
    x2 = epg.FSE_signal(angles_rad, phase_rad, TE, theta2['T1'], theta2['T2']) * (1 - exp(-(TR - T * TE)/theta2['T1'])) 

    return 0.5 * np.linalg.norm(x1, ord=2)**2 + 0.5 * np.linalg.norm(x2, ord=2)**2 - 0.5 * np.vdot(x1.ravel(), x2.ravel()) - 0.5 * np.vdot(x2.ravel(), x1.ravel())


def normalized_loss(theta1, theta2, angles_rad, phase_rad, TE, TR):
    T = len(angles_rad)
    x1 = epg.FSE_signal(angles_rad, phase_rad, TE, theta1['T1'], theta1['T2']) * (1 - exp(-(TR - T * TE)/theta1['T1']))
    x2 = epg.FSE_signal(angles_rad, phase_rad, TE, theta2['T1'], theta2['T2']) * (1 - exp(-(TR - T * TE)/theta2['T1']))

    x1 = x1 / np.linalg.norm(x1, ord=2)
    x2 = x2 / np.linalg.norm(x2, ord=2)

    return -0.5 * (np.vdot(x1.ravel(), x2.ravel()) + np.vdot(x2.ravel(), x1.ravel()))
    


def loss_prime(theta1, theta2, angles_rad, phase_rad, TE, TR, excitation_dict=None, inversion_dict=None):
    T = len(angles_rad)
    x1 = epg.FSE_signal(angles_rad, phase_rad, TE, theta1['T1'], theta1['T2'],
        excitation_dict=excitation_dict, inversion_dict=inversion_dict).ravel() * (1 - exp(-(TR - T * TE)/theta1['T1']))
    x2 = epg.FSE_signal(angles_rad, phase_rad, TE, theta2['T1'], theta2['T2'],
        excitation_dict=excitation_dict, inversion_dict=inversion_dict).ravel() * (1 - exp(-(TR - T * TE)/theta1['T1']))

    T = len(angles_rad)
    alpha_prime = np.zeros((T,))
    angle_e_prime = 0.
    angle_TI_prime = 0.

    for i in range(T):
        x1_prime = sig_prime_i(theta1, angles_rad, phase_rad, i).ravel() * (1 - exp(-(TR - T * TE)/theta1['T1']))
        x2_prime = sig_prime_i(theta2, angles_rad, phase_rad, i).ravel() * (1 - exp(-(TR - T * TE)/theta2['T1']))
        M1 = 0.5 * (np.vdot(x1, x1_prime) + np.vdot(x1_prime, x1))
        M2 = 0.5 * (np.vdot(x2, x2_prime) + np.vdot(x2_prime, x2))
        M3 = 0.5 * (np.vdot(x2_prime, x1) + np.vdot(x1, x2_prime))
        M4 = 0.5 * (np.vdot(x1_prime, x2) + np.vdot(x2, x1_prime))

        alpha_prime[i] = np.real(M1 + M2 - M3 - M4)

    return alpha_prime


def sig_prime_i(theta, angles_rad, phase_rad, idx, excitation_dict=None, inversion_dict=None):
    T1, T2 = get_params(theta)
    T = len(angles_rad)
    zi = np.hstack((np.array([[1],[1],[0]]), np.zeros((3, T))))

    z_prime = np.zeros((T, 1))

    for i in range(T):
        alpha = angles_rad[i]
        phi = phase_rad[i]
        if i < idx:
            zi = epg.FSE_TE(zi, alpha, phi, TE, T1, T2, noadd=True)
            z_prime[i] = 0
        elif i == idx:
            wi = epg.FSE_TE_prime(zi, alpha, phi, TE, T1, T2, noadd=True)
            z_prime[i] = np.real(wi[0,0])
        else:
            wi = epg.FSE_TE(wi, alpha, TE, phi, T1, T2, noadd=True, recovery=False)
            z_prime[i] = np.real(wi[0,0])

    return z_prime


def get_params(theta):
    return theta['T1'], theta['T2']


def numerical_gradient(theta1, theta2, angles_rad, phase_rad, TE, TR):
    initial_params = angles_rad
    num_grad = np.zeros(initial_params.shape)
    perturb = np.zeros(initial_params.shape)
    e = 1e-5

    for p in range(len(initial_params)):
        perturb[p] = e
        loss2 = loss(theta1, theta2, angles_rad + perturb, phase_rad, TE, TR)
        loss1 = loss(theta1, theta2, angles_rad - perturb, phase_rad, TE, TR)

        num_grad[p] = np.real(loss2 - loss1) / (2 * e)

        perturb[p] = 0

    return num_grad

def DEG2RAD(angle):
    return np.pi * angle / 180

def RAD2DEG(angle_rad):
    return 180 * angle_rad / np.pi

def read_angles(fliptable):
    f = open(fliptable, 'r')
    angles = []
    for line in f.readlines():
        angles.append(float(line))
    return np.array(angles)

def print_table(P1, P2, P3):
    print
    print '\tP1\tP2\tP3\nloss\t%3.3f\t%3.3f\t%3.3f\nnloss\t%3.3f\t%3.3f\t%3.3f\n' % (
            loss(theta1, theta2, P1.angles_rad, P1.phase_rad, TE, TR),
            loss(theta1, theta2, P2.angles_rad, P2.phase_rad, TE, TR),
            loss(theta1, theta2, P3.angles_rad, P3.phase_rad, TE, TR),
            normalized_loss(theta1, theta2, P1.angles_rad, P1.phase_rad, TE, TR),
            normalized_loss(theta1, theta2, P2.angles_rad, P2.phase_rad, TE, TR),
            normalized_loss(theta1, theta2, P3.angles_rad, P3.phase_rad, TE, TR)
            )



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True, precision=3)

    T1 = 1000e-3
    T2 = 200e-3

    TE = 50e-3
    TI = 0.
    TR = 1.4

    if len(sys.argv) > 1:
        T = int(sys.argv[1])
    else:
        T = 10

    angles = 150 * np.ones((T,))
    angles = read_angles('../data/flipangles.txt.408183520')

    TT = len(angles)
    if TT < T:
        T = TT
    else:
        angles = angles[:T]

    phases = np.zeros(angles.shape)

    angles_rad = DEG2RAD(angles)
    phase_rad = DEG2RAD(phases)

    S = epg.FSE_signal(angles_rad, phase_rad, TE, T1, T2)
    S2 = abs(S)

    theta1 = {'T1': 1000e-3, 'T2': 20e-3}
    theta2 = {'T1': 1000e-3, 'T2': 100e-3}

    t1 = time.time()
    NG = numerical_gradient(theta1, theta2, angles_rad, phase_rad, TE, TR)
    t2 = time.time()
    LP = loss_prime(theta1, theta2, angles_rad, phase_rad, TE, TR)
    t3 = time.time()

    NG_time = t2 - t1
    LP_time = t3 - t2

    print 'Numerical Gradient\t(%03.3f s)\t' % NG_time, NG
    print
    print 'Analytical Gradient\t(%03.3f s)\t' % LP_time, LP
    print
    print 'Error:', np.linalg.norm(NG - LP) / np.linalg.norm(NG)

    #plt.plot(TE*1000*np.arange(1, T+1), S2)
    #plt.xlabel('time (ms)')
    #plt.ylabel('signal')
    #plt.title('T1 = %.2f ms, T2 = %.2f ms' % (T1 * 1000, T2 * 1000))
    #plt.show()


    a = angles_rad
    #a = np.pi * np.ones((T,))
    a = None

    P1 = PulseTrain('angles_rand.mat', T, TE, TR, loss, loss_prime, angles_rad=a, verbose=True)
    #P1.load_state()
    P2 = PulseTrain('angles_180.mat', T, TE, TR, loss, loss_prime, angles_rad=np.pi * np.ones((T,)), verbose=True)
    P3 = PulseTrain('angles_vfa.mat', T, TE, TR, loss, loss_prime, angles_rad=angles_rad, verbose=True)

    print_table(P1, P2, P3)
    P1.train(theta1, theta2)
    print_table(P1, P2, P3)

    plt.figure(1)
    plt.clf()
    P1.plot_vals((theta1, theta2))

    plt.figure(2)
    plt.clf()
    P2.plot_vals((theta1, theta2))

    plt.figure(3)
    plt.clf()
    P3.plot_vals((theta1, theta2))

    plt.show()

    MAX_ANGLE = DEG2RAD(120)
    MIN_ANGLE = DEG2RAD(50)

