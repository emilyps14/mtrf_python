#!/bin/bash
today=`date '+%Y_%m_%d_%H_%M_%S'`;
outpath=$1;
i=$2;
submit_job -q <BATCH QUEUE> \
  -e <EMAIL> \
  -c 8 \
  -m 20 \
  -o <PATH TO MTRF_PYTHON>/mtrf_python/bash/cv_job_$i'_'$today'.log' \
  -n 'cv_'$i'_'$today \
  -x <PATH TO CONDA FOLDER>/.conda/envs/mtrf_python/bin/python <PATH TO MTRF_PYTHON>/mtrf_python/mtrf_python/scripts/run_cv_job_script.py $outpath

