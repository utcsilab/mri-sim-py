#!/usr/bin/env python


import sys
import cpmg_prop as cp
import matplotlib.pyplot as plt
import numpy as np

fname = sys.argv[1]
print(len(sys.argv))

# dummy values
P = cp.PulseTrain(fname, 10, 5e-3, 1, cp.loss, cp.loss_prime)

T1T2_vals = np.load('T1T2_vals_brain_K5.npy') * 1e-3
n_theta = T1T2_vals.shape[0]
thetas = []
for i in range(n_theta):
    T1, T2 = T1T2_vals[i,:]
    thetas.append({'T1': T1, 'T2': T2})

#thetas = [{'T1': np.mean(T1T2_vals[:,0]), 'T2': np.mean(T1T2_vals[:,1])}]
#thetas = [{'T1': 1, 'T2': .07}]

print(thetas)

leg_str = []
for fname in sys.argv[1:]:
    P.load_state(fname)
    P.plot_vals(thetas)
    P.compute_metrics(thetas)
    leg_str.append(fname[-27:-13])

plt.subplot(2,1,1)
plt.legend(leg_str)
plt.show()
