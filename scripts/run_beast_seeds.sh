#!/usr/bin/env bash
#SBATCH --array=1-5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=data/logs/%x-%A_%a.out

# Submits one SLURM job array of five seeds for every XML in data/xmls.
# Usage: scripts/run_beast_seeds.sh
#
# Run without SLURM, the script submits the arrays. SLURM then runs the script once per array task
# with SLURM_ARRAY_TASK_ID set, and each task runs BEAST on the XML in BEAST_XML with that seed.
# BEAST is started through the exec:exec goal of the pom. Extra maven flags can be passed with MVN_FLAGS, 
# the default is -q.

# submit

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    script=$(realpath "$0")
    cd "$(dirname "$script")/.."
    mkdir -p data/logs data/out

    for xml in data/xmls/coal_emat_*.xml; do
        sbatch --job-name="beast-$(basename "$xml" .xml)" --export=ALL,BEAST_XML="$xml" "$script"
    done

    for xml in data/xmls/bd_emat_*.xml; do
        sbatch --job-name="beast-$(basename "$xml" .xml)" --export=ALL,BEAST_XML="$xml" "$script"
    done

    exit 0
fi

# sbatch runs the copy of the script in the spool, so return to the directory the arrays were submitted from
cd "$SLURM_SUBMIT_DIR"

# setup

module load stack/2024-06 openjdk/25.0.2_10 maven/3.8.4

# run

seed=$SLURM_ARRAY_TASK_ID
echo "running $BEAST_XML with seed $seed"
mvn ${MVN_FLAGS:--q} exec:exec -Dbeast.args="-seed $seed $BEAST_XML -overwrite"
