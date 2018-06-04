#!/bin/bash

MAX_ITER=1000
STEP=0.1
ESP=5
#T1=1000
#T2=70
T1T2_vals_file="T1T2_vals_brain_K5.npy"
SOLVER="scipy"
SAVE_PARTIAL=1

ETL=$1

TRs=(500 1000 2000 10000 inf)
MAX_POWs=$(seq 30 3 300)

OUT_DIR=out
LOG_DIR=log

MAX_POW_prev="none"

for MAX_POW in ${MAX_POWs[@]} ; do
	for TR in ${TRs[@]} ; do 

		fname_prev="${T1T2_vals_file}_TE_${ESP}_TR_${TR}_ETL_${ETL}_POW_${MAX_POW_prev}"
		fname="${T1T2_vals_file}_TE_${ESP}_TR_${TR}_ETL_${ETL}_POW_${MAX_POW}"
		output_state="${OUT_DIR}/${fname}_state.pickle"
		output_angles="${OUT_DIR}/${fname}_angles.txt"
		input_angles="${OUT_DIR}/${fname_prev}_angles.txt"
		log="${LOG_DIR}/${fname}_log.txt"

		echo OMP_NUM_THREADS=1 python cpmg_prop.py --max_iter ${MAX_ITER} --max_power ${MAX_POW} --step ${STEP} --verbose --esp ${ESP} --etl ${ETL} --T1T2_vals ${T1T2_vals_file} --TR ${TR} --output_state ${output_state} --output_angles ${output_angles} --input_angles ${input_angles} --save_partial ${SAVE_PARTIAL} --solver ${SOLVER} \> $log 2\>\&1

	done

	MAX_POW_prev=${MAX_POW}
done
