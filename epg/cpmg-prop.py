#!/usr/bin/env python

import numpy as np
from numpy import pi, cos, sin, exp, conj
from warnings import warn
import epgcpmg as epg
import time
import sys
from argparse import ArgumentParser
import pickle


class PulseTrain:
    def __init__(self, state_file, T, TE, TR, loss_fun, loss_fun_prime, angles_rad=None, verbose=False, step=.01, max_iter=100, prox_fun=None):
        self.state_file = state_file
        self.T = T
        self.TE = TE
        self.TR = TR
        self.loss_fun = loss_fun
        self.loss_fun_prime = loss_fun_prime
        self.prox_fun = prox_fun
        self.max_iter = max_iter
        self.step = step
        self.verbose = verbose

        self.reset()

        if angles_rad is not None:
            self.set_angles_rad(angles_rad)

    def set_angles_rad(self, angles_rad):
        T = len(angles_rad)
        if T < self.T:
            self.angles_rad = np.hstack((angles_rad, np.zeros((self.T-T))))
        else:
            self.angles_rad = angles_rad[:self.T]

    def reset(self):
        self.angles_rad = DEG2RAD(50 + (120 - 50) * np.random.rand(self.T))
        self.loss = []
        self.sqrt_max_power = []

    def save_state(self, filename=None):
        state = {
                'angles_rad': self.angles_rad,
                'loss': self.loss,
                'sqrt_max_power': self.sqrt_max_power,
                'max_iter': self.max_iter,
                'step': self.step,
                'T': self.T,
                'TE': self.TE,
                'TR': self.TR,
                'verbose': self.verbose,
                }
        if filename is None:
            filename = self.state_file

        pickle_out = open(filename, 'wb')
        pickle.dump(state, pickle_out)
        pickle_out.close()

    def load_state(self, filename=None):
        if filename is None:
            filename = self.state_file

        print('loading state from file {}'.format(filename))

        pickle_in = open(filename, 'rb')
        state = pickle.load(pickle_in)
        pickle_in.close()

        self.angles_rad = state['angles_rad']
        self.loss = state['loss']
        self.sqrt_max_power = state['sqrt_max_power']
        self.max_iter = state['max_iter']
        self.step = state['step']
        self.T = state['T']
        self.TE = state['TE']
        self.TR = state['TR']
        self.verbose = state['verbose']


    def train(self, theta1):
        for i in range(self.max_iter):
            angles_prime = self.loss_fun_prime(theta1, self.angles_rad, self.TE, self.TR)
            self.angles_rad = self.angles_rad + self.step * angles_prime
            if self.prox_fun is not None:
                self.angles_rad = self.prox_fun(theta1, self.angles_rad, self.step)
            self.sqrt_max_power.append(np.linalg.norm(self.angles_rad))

            self.loss.append(self.loss_fun(theta1, self.angles_rad, self.TE, self.TR))
            str = '%d\t%3.5f\t%3.5f' % (i, self.loss[-1], self.sqrt_max_power[-1])
            self.print_verbose(str)

    def print_verbose(self, str):
        if self.verbose:
            print(str, RAD2DEG(self.angles_rad))

    def plot_vals(self, thetas):
        plt.subplot(2,1,1)
        plt.plot(range(self.T), RAD2DEG(self.angles_rad), 'o-')
        plt.xlim((0,self.T))
        plt.subplot(2,1,2)
        for theta in thetas:
            plt.plot(range(self.T), epg.FSE_signal(self.angles_rad, self.TE, theta['T1'], theta['T2']))
        plt.xlim((0,self.T))
        plt.ylim((0,1))

    def forward(self, theta):
        return epg.FSE_signal(self.angles_rad, TE, theta['T1'], theta['T2']).ravel()


def loss(theta1, angles_rad, TE, TR):
    T = len(angles_rad)
    
    x1 = epg.FSE_signal(angles_rad, TE, theta1['T1'], theta1['T2']) * (1 - exp(-(TR - T * TE)/theta1['T1']))

    return 0.5 * np.linalg.norm(x1, ord=2)**2

def normalized_loss(theta1, angles_rad, TE, TR):
    T = len(angles_rad)
    x1 = epg.FSE_signal(angles_rad, TE, theta1['T1'], theta1['T2']) * (1 - exp(-(TR - T * TE)/theta1['T1']))

    x1 = x1 / np.linalg.norm(x1, ord=2)

    return np.dot(x1.ravel(), x1.ravel())


def loss_prime(theta1, angles_rad, TE, TR):
    T = len(angles_rad)
    x1 = epg.FSE_signal(angles_rad, TE, theta1['T1'], theta1['T2']).ravel() * (1 - exp(-(TR - T * TE)/theta1['T1']))

    T = len(angles_rad)
    alpha_prime = np.zeros((T,))

    for i in range(T):
        x1_prime = epg.FSE_signal_prime_alpha_idx(angles_rad, TE, theta1['T1'], theta1['T2'], i).ravel() * (1 - exp(-(TR - T * TE)/theta1['T1']))
        alpha_prime[i] = np.dot(x1, x1_prime)

    return alpha_prime


def get_params(theta):
    return theta['T1'], theta['T2']

def prox_fun(theta, angles_rad, mu):
    angles_rad[angles_rad<0] = 0
    angles_rad[angles_rad>np.pi] = np.pi
    A = theta['sqrt_max_power']
    q1 = np.linalg.norm(angles_rad)
    if q1 > A:
        return angles_rad / q1 * A
    else:
        return angles_rad


def numerical_gradient(theta1, angles_rad, TE, TR):
    initial_params = angles_rad
    num_grad = np.zeros(initial_params.shape)
    perturb = np.zeros(initial_params.shape)
    e = 1e-5

    for p in range(len(initial_params)):
        perturb[p] = e
        loss2 = loss(theta1, angles_rad + perturb, TE, TR)
        loss1 = loss(theta1, angles_rad - perturb, TE, TR)

        num_grad[p] = (loss2 - loss1) / (2 * e)

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
    f.close()
    return np.array(angles)

def write_angles(fliptable, angles):
    f = open(fliptable, 'w')
    for a in angles:
        f.write('%f\n' % a)
    f.close()

#def print_table(P1, P2, P3):
    #print
    #print '\tP1\tP2\tP3\nloss\t%3.3f\t%3.3f\t%3.3f\nnloss\t%3.3f\t%3.3f\t%3.3f\n' % (
            #loss(theta1, P1.angles_rad, TE, TR),
            #loss(theta1, P2.angles_rad, TE, TR),
            #loss(theta1, P3.angles_rad, TE, TR),
            #normalized_loss(theta1, P1.angles_rad, TE, TR),
            #normalized_loss(theta1, P2.angles_rad, TE, TR),
            #normalized_loss(theta1, P3.angles_rad, TE, TR)
            #)


def get_usage_str():
    return "usage: %(prog)s [options]"

def get_version_str():
    return "Version 0.4"

def get_description_str():
    return """EPG CMPG back-propagation.
    Jon Tamir <jtamir@eecs.berkeley.edu>"""

def parser_defaults():
    d = {
            'max_iter': 100,
            'verbose': False,
            'step': .1,
            'max_power': None,
            'esp': 5,
            'etl': 20,
            'TR': 1500,
            'T1': 1000,
            'T2': 100,
            'input_state_file': None,
            'output_state_file': None,
            'input_angles_file': None,
            'output_angles_file': None,
            }
    return d


def get_parser(usage_str, description_str, version_str, parser_defaults):
    parser = ArgumentParser(usage=usage_str, description=description_str)

    parser.add_argument('--max_iter', action='store', dest='max_iter', type=int, help='max iter')
    parser.add_argument('--step', action='store', dest='step', type=float, help='step size')
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')
    parser.add_argument('--max_power', action='store', dest='max_power', type=float, help='max power constraint')
    parser.add_argument('--esp', action='store', dest='esp', type=float, help='echo spacing in ms')
    parser.add_argument('--etl', action='store', dest='etl', type=int, help='echo train length')
    parser.add_argument('--T1', action='store', dest='T1', type=float, help='T1 in ms')
    parser.add_argument('--T2', action='store', dest='T2', type=float, help='T2 in ms')
    parser.add_argument('--TR', action='store', dest='TR', type=float, help='TR in ms')
    parser.add_argument('--input_state', action='store', type=str, dest='input_state_file', help='initialize state from pickle file')
    parser.add_argument('--output_state', action='store', type=str, dest='output_state_file', help='save state to pickle file')
    parser.add_argument('--input_angles', action='store', type=str, dest='input_angles_file', help='initialize angles from txt file')
    parser.add_argument('--output_angles', action='store', type=str, dest='output_angles_file', help='save angles to txt file')

    parser.set_defaults(**parser_defaults)

    return parser


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    parser = get_parser(get_usage_str(), get_description_str(), get_version_str(), parser_defaults())

    print(get_description_str())
    print(get_version_str())

    np.set_printoptions(suppress=True, precision=3)

    args = parser.parse_args()
    print(args)

    T1 = args.T1 * 1e-3
    T2 = args.T2 * 1e-3
    max_power = args.max_power

    TE = args.esp * 1e-3
    TR = args.TR * 1e-3
    ETL = args.etl

    step = args.step
    max_iter = args.max_iter
    verbose = args.verbose

    input_angles_file = args.input_angles_file
    output_angles_file = args.output_angles_file

    input_state_file = args.input_state_file
    output_state_file = args.output_state_file



    if input_angles_file is None:
        angles = 120  * np.ones((ETL,))
    else:
        angles = read_angles(input_angles_file)

    TT = len(angles)
    if TT < ETL:
        warn('warning: number of input flip angles ({1}) less than ETL ({2}), setting ETL to {1}'.format(TT, ETL))
        ETL = TT
    elif TT > ETL:
        warn('warning: number of input flip angles ({1}) greater than ETL ({2}), clipping flip angles'.format(TT, ETL))
        angles = angles[:ETL]

    angles_rad = DEG2RAD(angles)

    #S = epg.FSE_signal(angles_rad, TE, T1, T2)
    #S2 = abs(S)

    #max_power = ETL * (120 * np.pi / 180)**2

    if max_power is None:
        sqrt_max_power = None
        prox_fun = None
    else:
        sqrt_max_power = np.sqrt(max_power)
        print('max power: {}'.format(max_power))
        print('sqrt max power: {}'.format(sqrt_max_power))

    theta1 = {'T1': T1, 'T2': T2, 'sqrt_max_power': sqrt_max_power}

    t1 = time.time()
    NG = numerical_gradient(theta1, angles_rad, TE, TR)
    t2 = time.time()
    LP = loss_prime(theta1, angles_rad, TE, TR)
    t3 = time.time()

    NG_time = t2 - t1
    LP_time = t3 - t2

    print('Numerical Gradient\t(%03.3f s)\t' % NG_time, NG)
    print()
    print('Analytical Gradient\t(%03.3f s)\t' % LP_time, LP)
    print()
    print('Error:', np.linalg.norm(NG - LP) / np.linalg.norm(NG))

    #plt.plot(TE*1000*np.arange(1, ETL+1), S2)
    #plt.xlabel('time (ms)')
    #plt.ylabel('signal')
    #plt.title('T1 = %.2f ms, T2 = %.2f ms' % (T1 * 1000, T2 * 1000))
    #plt.show()


    P = PulseTrain(input_state_file, ETL, TE, TR, loss, loss_prime, angles_rad=angles_rad, verbose=verbose, max_iter=max_iter, step=step, prox_fun=prox_fun)

    if input_state_file is not None:
        P.load_state(input_state_file)

    P.train(theta1)

    if output_state_file is not None:
        P.save_state(output_state_file)

    if output_angles_file is not None:
        write_angles(output_angles_file, RAD2DEG(P.angles_rad))
    #print_table(P1, P2, P3)

    #plt.figure(1)
    #plt.clf()
    #P.plot_vals((theta1))

    #plt.figure(2)
    #plt.clf()
    #P2.plot_vals((theta1))

    #plt.figure(3)
    #plt.clf()
    #P3.plot_vals((theta1))

    #plt.show()

    MAX_ANGLE = DEG2RAD(120)
    MIN_ANGLE = DEG2RAD(50)

