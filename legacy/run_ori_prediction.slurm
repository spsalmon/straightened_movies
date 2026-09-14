#!/bin/bash

#SBATCH --job-name="ori"
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --mem=48GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


# As a rough approximation for running time, in a test run with 32 CPUs, 150_000 str_images across 347 Points were processed in about 75min.
# N.B. the images are processed in parallel grouped by point, so if there are e.g. only 10 points, adding CPUs beyond 10 will NOT speed up the script


CURRENT_TIME="`date "+%Y-%m-%d %H:%M:%S"`"
echo "Start time:" $CURRENT_TIME


# For changing input/output, check the ori_align.py file

~/.local/bin/micromamba run \
    -n towbintools \
    python3 \
    -m ori_align



CURRENT_TIME="`date "+%Y-%m-%d %H:%M:%S"`"
echo "End time:" $CURRENT_TIME