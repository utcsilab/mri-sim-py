#!/bin/bash

jobs_dir=$1

echo "loading jobs from $jobs_dir"
echo "parallel jobs: ${OMP_NUM_THREADS}"

echo "start time: $(date)"

ls $jobs_dir/* | xargs -n1 -I{} -P ${OMP_NUM_THREADS} bash -c "{}"

echo "end time: $(date)"
