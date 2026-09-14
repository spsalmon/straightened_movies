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

EXPERIMENT="/mnt/towbin.data/shared/USER/EXPERIMENT"

~/.local/bin/micromamba run -n towbintools python3 \
    "$(dirname "$0")/straightened_movies.py" \
    --filemap "${EXPERIMENT}/analysis/report/analysis_filemap.csv" \
    --straightened_column "analysis/ch1_ch2_raw_str" \
    --orientation_channel 1 \
    --qc_column "ch2_seg_str_qc" \
    --atlas "/mnt/towbin.data/shared/bgusev/atlas/dark_germline/atlas.csv" \
    --atlas_channel 1 \
    --movie_columns All \
    --alignment left \
    --points All \
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
