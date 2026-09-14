#!/bin/bash

#SBATCH --job-name="movies"
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=48GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


# Time taken heavily depends on how under load the cluster is since making movies is a very I/O heavy task - all relevant images need to be loaded, put together, and saved as a single stack
# The actual computational heavy lifting has already been done by ori_align.py script

CURRENT_TIME="`date "+%Y-%m-%d %H:%M:%S"`"
echo "Start time:" $CURRENT_TIME


# For changing input/output, check the ori_align.py file

~/.local/bin/micromamba run \
    -n towbintools \
    python3 \
    -m movie_generation



CURRENT_TIME="`date "+%Y-%m-%d %H:%M:%S"`"
echo "End time:" $CURRENT_TIME