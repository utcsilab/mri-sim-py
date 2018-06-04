#!/usr/bin/env python


import sys
import cpmg_prop as cp
import matplotlib.pyplot as plt
import numpy as np

state_in_fname = sys.argv[1]
angles_fname = sys.argv[2]
state_out_fname = sys.argv[3]

# dummy values
P = cp.PulseTrain(state_in_fname, 10, 5e-3, 1, cp.loss, cp.loss_prime)

P.load_state(state_in_fname)
angles = cp.read_angles(angles_fname)

#assert P.T == len(angles), "angles and state files must be same length"
P.angles_rad = cp.DEG2RAD(angles)
P.T = len(P.angles_rad)

print('power: {}'.format(cp.calc_power(P.angles_rad)))

P.save_state(state_out_fname)

sys.exit(0)
