#!/usr/bin/env bash
#SBATCH --job-name=plots
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/%x-%j.out

# Makes the plots of the BEAST runs by running `make plots`.
# Usage: sbatch scripts/run_plots.sh (from the repository root)

# sbatch runs the copy of the script in the spool, so return to the directory the job was submitted from
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# run

make plots
