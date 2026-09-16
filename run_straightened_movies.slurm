#!/bin/bash

#SBATCH --job-name="str_movies"
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Orientation prediction dominates the runtime and parallelises over points, so
# there is nothing to gain from more CPUs than there are points. Movie assembly is
# I/O bound: every straightened image is read back and written out as one stack.
# Re-runs over an experiment that already has an orientations.csv skip prediction
# entirely, which makes trying a different --alignment cheap.

echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

EXPERIMENT="/mnt/towbin.data/shared/kstojanovski/20240202_Orca_10x_yap-1del_col-10-tir_wBT160-186-310-337-380-393_25C_20240202_171239_051"
ANALYSIS_DIR="analysis_sacha"
~/.local/bin/micromamba run -n towbintools python3 \
    "$(dirname "$0")/straightened_movies.py" \
    --filemap "${EXPERIMENT}/${ANALYSIS_DIR}/report/analysis_filemap.csv" \
    --straightened_column "${ANALYSIS_DIR}/ch2_raw_str" \
    --orientation_channel 0 \
    --qc_column "ch2_seg_str_qc" \
    --atlas "/mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv" \
    --atlas_channel 1 \
    --movie_columns "${ANALYSIS_DIR}/ch2_raw_str" \
    --alignment center \
    --points 0-3 \
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
