#!/usr/bin/env python

import numpy as np
from numpy import pi, cos, sin, exp, conj
import scipy.optimize
from warnings import warn
import epgcpmg as epg
import time
import sys
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt




class PulseTrain:
    def __init__(self, state_file, T, TE, TR, loss_fun, loss_fun_prime, angles_rad=None, verbose=False, step=.01, max_iter=100, prox_fun=None, prox_fun_prime=None, save_partial=100, solver='pgd', solver_opts={'step':.01, 'max_iter': 100}, min_flip_rad=0, max_flip_rad=np.pi):
        self.state_file = state_file
        self.T = T
        self.TE = TE
        self.TR = TR
        self.loss_fun = loss_fun
        self.loss_fun_prime = loss_fun_prime
        self.prox_fun = prox_fun
        self.prox_fun_prime = prox_fun_prime
        self.max_iter = max_iter
        self.step = step
        self.verbose = verbose
        self.save_partial = save_partial
        self.solver = solver
        self.solver_opts = solver_opts
        self.min_flip_rad=min_flip_rad,
        self.max_flip_rad=max_flip_rad

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
        self.angles_rad = DEG2RAD(50 + (180 - 50) * np.random.rand(self.T))
        self.loss = []
        self.sqrt_max_power = []

    def save_state(self, filename=None):
        state = {
                'angles_rad': self.angles_rad,
                'loss': self.loss,
                'sqrt_max_power': self.sqrt_max_power,
                'solver_opts': self.solver_opts,
                'solver': self.solver,
                'T': self.T,
                'TE': self.TE,
                'TR': self.TR,
                'verbose': self.verbose,
                'save_partial': self.save_partial,
                'min_flip_rad': self.min_flip_rad,
                'max_flip_rad': self.max_flip_rad,
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

        try:
            self.angles_rad = state['angles_rad']
        except:
            self.angles_rad = None
        try:
            self.loss = state['loss']
        except:
            self.loss = None
        try:
            self.sqrt_max_power = state['sqrt_max_power']
        except:
            self.sqrt_max_power = None 
        try:
            self.solver = state['solver']
        except:
            self.solver = None
        try:
            self.solver_opts = state['solver_opts']
        except:
            self.solver_opts = None
        try:
            self.T = state['T']
        except:
            self.T = None
        try:
            self.TE = state['TE']
        except:
            self.TE = None
        try:
            self.TR = state['TR']
        except:
            self.TR = None
        try:
            self.verbose = state['verbose']
        except:
            self.verbose = None
        try:
            self.save_partial = state['save_partial']
        except:
            self.save_partial = None
        try:
            self.min_flip_rad = state['min_flip_rad']
        except:
            self.min_flip_rad = None
        try:
            self.max_flip_rad = state['max_flip_rad']
        except:
            self.max_flip_rad = None


    def train(self, thetas):
        tic = time.time()

        if self.solver == 'pgd':
            max_iter = self.solver_opts['max_iter']
            step = self.solver_opts['step']
            for i in range(max_iter):
                angles_prime = self.loss_fun_prime(thetas, self.angles_rad, self.TE, self.TR)
                self.angles_rad = self.angles_rad + step * angles_prime
                if self.prox_fun is not None:
                    self.angles_rad = self.prox_fun(thetas, self.angles_rad, self.step)
                self.angles_rad[self.angles_rad < self.min_flip_rad] = self.min_flip_rad
                self.angles_rad[self.angles_rad > self.max_flip_rad] = self.max_flip_rad
                if i % self.save_partial == 0:
                    self.save_state(self.state_file)
                    self.sqrt_max_power.append(np.linalg.norm(self.angles_rad))
                    self.loss.append(self.loss_fun(thetas, self.angles_rad, self.TE, self.TR))
                    str = '%d\t%3.5f\t%3.5f' % (i, self.loss[-1], self.sqrt_max_power[-1])
                    self.print_verbose(str)

        elif self.solver == 'scipy':

            def myloss(x, info):
                self.angles_rad = x
                info['nfev'] += 1
                if info['nfev'] % self.save_partial == 0:
                    self.save_state(self.state_file)
                    self.sqrt_max_power.append(np.linalg.norm(self.angles_rad))
                    self.loss.append(self.loss_fun(thetas, self.angles_rad, self.TE, self.TR))
                    str = '%d\t%3.5f\t%3.5f' % (info['nfev'], self.loss[-1], self.sqrt_max_power[-1])
                    self.print_verbose(str)
                return -self.loss_fun(thetas, x, self.TE, self.TR)

            res = scipy.optimize.minimize(myloss, self.angles_rad,
                    args=({'nfev': 0},),
                    jac=lambda x, y: -self.loss_fun_prime(thetas, x, self.TE, self.TR),
                    bounds=np.array([self.min_flip_rad * np.ones((P.T,)), self.max_flip_rad * np.ones((P.T,))]).T,
                    constraints=({'type': 'ineq', 'fun': lambda x: thetas[0]['sqrt_max_power'] - np.linalg.norm(x)}),
                    options={'maxiter': self.solver_opts['max_iter']}, method='SLSQP')
            if self.verbose:
                print(res)
            self.angles_rad = res.x
        else:
            print('ERROR: {} not a recognized solver'.format(self.solver))
            sys.exit(-1)
        toc = time.time()
        if verbose:
            print('finished optimization in {:.2f} s'.format(toc - tic))

    def print_verbose(self, str):
        if self.verbose:
            print(str, RAD2DEG(self.angles_rad))

    def plot_vals(self, thetas):
        plt.subplot(2,1,1)
        plt.plot(range(self.T), RAD2DEG(self.angles_rad), 'o-')
        plt.title('ETL={} POW={:.1f} MAX={:.0f} MIN={:.0f}'.format(
            self.T, calc_power(self.angles_rad), RAD2DEG(np.max(self.angles_rad)), RAD2DEG(np.min(self.angles_rad))))
        plt.xlim((0, self.T))
        #plt.ylim((np.max((0,.9*np.min(RAD2DEG(self.angles_rad))), 180)))
        plt.ylim((.5*np.min(RAD2DEG(self.angles_rad)), 180))
        plt.ylabel('flip angles (deg)')
        plt.subplot(2,1,2)
        #leg_str = []
        for theta in thetas:
            plt.plot(range(self.T), epg.FSE_signal(self.angles_rad, self.TE, theta['T1'], theta['T2']) * (1 - exp(-(self.TR - self.T * self.TE)/theta['T1'])))
            #leg_str.append('T1/T2={:.0f}/{:.0f}'.format(1000*theta['T1'], 1000*theta['T2']))
        #plt.legend(leg_str)
        plt.xlim((0,self.T))
        plt.ylim((0, 1.))
        plt.ylabel('signal level')

    def compute_metrics(self, thetas):
        flip_power = calc_power(self.angles_rad)
        print('max\tmin\tpow')
        print('{}\t{}\t{}'.format(RAD2DEG(np.max(self.angles_rad)), RAD2DEG(np.min(self.angles_rad)), flip_power))
        print('SNR: {}'.format(calc_SNR(self.loss_fun(thetas, self.angles_rad, self.TE, self.TR))))
        #for i, theta in enumerate(thetas):
            #print('SNR theta {}: {}'.format(i, calc_SNR(self.loss_fun([theta], self.angles_rad, self.TE, self.TR))))



def loss(thetas, angles_rad, TE, TR):
    T = len(angles_rad)
    
    l = 0
    for theta in thetas:
        x1 = epg.FSE_signal(angles_rad, TE, theta['T1'], theta['T2']) * (1 - exp(-(TR - T * TE)/theta['T1']))
        l += 0.5 * np.dot(x1.ravel(), x1.ravel())

    return l

def loss_prime(thetas, angles_rad, TE, TR):

    T = len(angles_rad)
    alpha_prime = np.zeros((T,))

    for theta in thetas:

        x1 = epg.FSE_signal(angles_rad, TE, theta['T1'], theta['T2']).ravel() * (1 - exp(-(TR - T * TE)/theta['T1']))

        for i in range(T):

            x1_prime = epg.FSE_signal_prime_alpha_idx(angles_rad, TE, theta['T1'], theta['T2'], i).ravel() * (1 - exp(-(TR - T * TE)/theta['T1']))
            alpha_prime[i] += np.dot(x1, x1_prime)

    return alpha_prime


def get_params(theta):
    return theta['T1'], theta['T2']

def prox_fun(theta, angles_rad, mu):
    A = theta['sqrt_max_power']
    q1 = np.linalg.norm(angles_rad)
    if q1 > A:
        return angles_rad / q1 * A
    else:
        return angles_rad

def calc_power(angles_rad):
    return np.linalg.norm(angles_rad)**2

def calc_SNR(sig):
    return np.linalg.norm(sig)




def numerical_gradient(loss_fun, thetas, angles_rad, TE, TR):
    initial_params = angles_rad
    num_grad = np.zeros(initial_params.shape)
    perturb = np.zeros(initial_params.shape)
    e = 1e-5

    for p in range(len(initial_params)):
        perturb[p] = e
        loss2 = loss_fun(thetas, angles_rad + perturb, TE, TR)
        loss1 = loss_fun(thetas, angles_rad - perturb, TE, TR)

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
            'min_flip': 0,
            'max_flip': 180,
            'esp': 5,
            'etl': 20,
            'TR': 1500,
            'T1': 1000,
            'T2': 100,
            'T1T2_vals_file': None,
            'input_state_file': None,
            'output_state_file': None,
            'input_angles_file': None,
            'output_angles_file': None,
            'save_partial': 100,
            'solver': 'pgd',
            }
    return d


def get_parser(usage_str, description_str, version_str, parser_defaults):
    parser = ArgumentParser(usage=usage_str, description=description_str)

    parser.add_argument('--max_iter', action='store', dest='max_iter', type=int, help='max iter')
    parser.add_argument('--step', action='store', dest='step', type=float, help='step size')
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')
    parser.add_argument('--max_power', action='store', dest='max_power', type=float, help='max power constraint')
    parser.add_argument('--min_flip', action='store', dest='min_flip', type=float, help='min flip angle in deg')
    parser.add_argument('--max_flip', action='store', dest='max_flip', type=float, help='max flip angle in deg')
    parser.add_argument('--esp', action='store', dest='esp', type=float, help='echo spacing in ms')
    parser.add_argument('--etl', action='store', dest='etl', type=int, help='echo train length')
    parser.add_argument('--T1', action='store', dest='T1', type=float, help='T1 in ms')
    parser.add_argument('--T2', action='store', dest='T2', type=float, help='T2 in ms')
    parser.add_argument('--T1T2_vals', action='store', dest='T1T2_vals_file', type=str, help='use T1 and T2 values from T1T2_vals.npy (in ms)')
    parser.add_argument('--TR', action='store', dest='TR', type=float, help='TR in ms')
    parser.add_argument('--input_state', action='store', type=str, dest='input_state_file', help='initialize state from pickle file')
    parser.add_argument('--output_state', action='store', type=str, dest='output_state_file', help='save state to pickle file')
    parser.add_argument('--input_angles', action='store', type=str, dest='input_angles_file', help='initialize angles from txt file')
    parser.add_argument('--output_angles', action='store', type=str, dest='output_angles_file', help='save angles to txt file')
    parser.add_argument('--save_partial', action='store', type=int, dest='save_partial', help='save state every <int> epochs')
    parser.add_argument('--solver', action='store', type=str, dest='solver', help='solver type (pgd -- prox grad desc, scipy -- scipy optimizer')

    parser.set_defaults(**parser_defaults)

    return parser


if __name__ == "__main__":
    parser = get_parser(get_usage_str(), get_description_str(), get_version_str(), parser_defaults())

    print(get_description_str())
    print(get_version_str())

    np.set_printoptions(suppress=True, precision=3)

    args = parser.parse_args()
    print(args)

    T1 = args.T1 * 1e-3
    T2 = args.T2 * 1e-3
    max_power = args.max_power
    min_flip = args.min_flip
    max_flip = args.max_flip

    TE = args.esp * 1e-3
    TR = args.TR * 1e-3
    ETL = args.etl

    solver = args.solver
    step = args.step
    max_iter = args.max_iter
    save_partial = args.save_partial
    verbose = args.verbose

    input_angles_file = args.input_angles_file
    output_angles_file = args.output_angles_file

    input_state_file = args.input_state_file
    output_state_file = args.output_state_file

    T1T2_vals_file = args.T1T2_vals_file

    if T1T2_vals_file is not None:
        if verbose:
            print('Using T1,T2 values from {}'.format(T1T2_vals_file))
        T1T2_vals = np.load(T1T2_vals_file) * 1e-3
    else:
        T1T2_vals = np.array([[T1], [T2]]).T

    n_theta = T1T2_vals.shape[0]


    if input_angles_file is not None:
        try:
            angles = read_angles(input_angles_file)
        except:
            warn('warning: cannot read from {}'.format(input_angles_file))
            angles = RAD2DEG(np.sqrt(max_power / ETL)) * np.ones((ETL,))
    else:
        angles = RAD2DEG(np.sqrt(max_power / ETL)) * np.ones((ETL,))

    TT = len(angles)
    if TT < ETL:
        warn('warning: number of input flip angles ({0}) less than ETL ({1}), setting ETL to {0}'.format(TT, ETL))
        ETL = TT
    elif TT > ETL:
        warn('warning: number of input flip angles ({0}) greater than ETL ({1}), clipping flip angles'.format(TT, ETL))
        angles = angles[:ETL]

    angles_rad = DEG2RAD(angles)

    if max_power is None:
        sqrt_max_power = None
        prox_fun = None
    else:
        sqrt_max_power = np.sqrt(max_power)
        print('max power: {}'.format(max_power))
        print('sqrt max power: {}'.format(sqrt_max_power))

    thetas = []
    for i in range(n_theta):
        T1, T2 = T1T2_vals[i,:]
        thetas.append({'T1': T1, 'T2': T2, 'sqrt_max_power': sqrt_max_power})

    if verbose:
        print(thetas)


    t1 = time.time()
    NG = numerical_gradient(loss, thetas, angles_rad, TE, TR)
    t2 = time.time()
    LP = loss_prime(thetas, angles_rad, TE, TR)
    t3 = time.time()

    NG_time = t2 - t1
    LP_time = t3 - t2

    print('Numerical Gradient\t(%03.3f s)\t' % NG_time, NG)
    print()
    print('Analytical Gradient\t(%03.3f s)\t' % LP_time, LP)
    print()
    print('Error:', np.linalg.norm(NG - LP) / np.linalg.norm(NG))

    solver_opts = {'max_iter': max_iter, 'step': step}

    P = PulseTrain(output_state_file, ETL, TE, TR, loss, loss_prime, angles_rad=angles_rad, verbose=verbose, solver=solver, solver_opts=solver_opts, prox_fun=prox_fun, save_partial=save_partial, min_flip_rad=DEG2RAD(min_flip), max_flip_rad=DEG2RAD(max_flip))

    if input_state_file is not None:
        P.load_state(input_state_file)

    P.train(thetas)

    if output_state_file is not None:
        P.save_state(output_state_file)

    if output_angles_file is not None:
        write_angles(output_angles_file, RAD2DEG(P.angles_rad))
