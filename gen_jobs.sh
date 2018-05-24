#!/bin/bash

MAX_ITER=10000
STEP=0.1
ESP=5
T1=1000
T2=100
TR=10000

ETLs=$(seq 30 140)
MAX_POWs=$(seq 30 300)

OUT_DIR=out
LOG_DIR=log

for ETL in ${ETLs[@]} ; do 
	for MAX_POW in ${MAX_POWs[@]} ; do

		fname="T1_${T1}_T2_${T2}_TE_${ESP}_TR_${TR}_ETL_${ETL}_POW_${MAX_POW}"
		output_state="${OUT_DIR}/${fname}_state.pickle"
		output_angles="${OUT_DIR}/${fname}_angles.txt"
		log="${LOG_DIR}/${fname}_log.txt"

		echo python cpmg_prop.py --max_iter ${MAX_ITER} --max_power ${MAX_POW} --step ${STEP} --verbose --esp ${ESP} --etl ${ETL} --T1 ${T1} --T2 ${T2} --TR ${TR} --output_state ${output_state} --output_angles ${output_angles} \> $log
	done
done
