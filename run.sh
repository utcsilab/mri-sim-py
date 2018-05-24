#!/bin/bash

jobs_file=$1

echo "loading jobs from $jobs_file"
echo "parallel jobs: ${OMP_NUM_THREADS}"

echo "start time: $(date)"

cat $jobs_file | xargs -n1 -I{} -P ${OMP_NUM_THREADS}  bash -c "{}"

echo "end time: $(date)"
