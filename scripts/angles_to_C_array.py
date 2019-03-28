#!/usr/bin/env python

import sys
import os
import numpy as np

max_pows = np.arange(30, 303, 3)
etls = np.arange(30, 141)
trs = np.array([500, 2000, 10000])

file_out = sys.argv[1]


fo = open(file_out, 'w')

for tr in trs:
    for etl in etls:
        base_name = 'optimized_angles/T1T2_vals_brain_K5.npy_TE_5_TR_{}_ETL_{}'.format(tr, etl)
        for pow in max_pows:
            filename = '{}_POW_{}_angles.txt'.format(base_name, pow)
            angles = np.loadtxt(filename)
            fo.write('float flip_opt_TR_{}_ETL_{}_POW_{}[{}] = {{'.format(tr, etl, pow, etl))
            for angle in angles[:-1]:
                fo.write('{:5.1f}, '.format(angle))
            fo.write('{:5.1f}}};\n'.format(angles[-1]))

        fo.write('float* flip_opt_TR_{}_ETL_{}[{}] = {{'.format(tr, etl, len(max_pows)))
        for pow in max_pows[:-1]:
            fo.write('flip_opt_TR_{}_ETL_{}_POW_{}, '.format(tr, etl, pow))
        fo.write('flip_opt_TR_{}_ETL_{}_POW_{}}};\n '.format(tr, etl, max_pows[-1]))
        fo.write('\n')

    fo.write('float** flip_opt_TR_{}[{}] = {{'.format(tr, len(etls)))
    for etl in etls[:-1]:
        fo.write('flip_opt_TR_{}_ETL_{}, '.format(tr, etl))
    fo.write('flip_opt_TR_{}_ETL_{}}};\n '.format(tr, etls[-1]))
    fo.write('\n')

    fo.write('\n')

fo.close()
