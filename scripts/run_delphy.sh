#!/usr/bin/env bash
set -euo pipefail

# Runs Delphy on each FASTA file in FASTAS (relative to data/fasta) with an exponential coalescent,
# no site rate heterogeneity and Delphy's default priors and initial values. Each run takes 1,000,000
# steps per tip (ten times Delphy's default) and logs every 100,000 steps.
# Usage: scripts/run_delphy.sh [seed]
#
# The Delphy binary can be overridden with DELPHY, the default is ../delphy/bin/delphy. Extra Delphy
# flags (e.g. --v0-threads) can be passed with DELPHY_FLAGS.

FASTAS=(
    h3n2.fasta
)

cd "$(dirname "$(realpath "$0")")/.."

delphy=${DELPHY:-../delphy/bin/delphy}
seed=${1:-1}
out=data/out
mkdir -p "$out"

# run

for fasta in "${FASTAS[@]}"; do
    name=coal_delphy_$(basename "$fasta" .fasta)-$seed
    tips=$(grep -c '^>' "data/fasta/$fasta")
    steps=$((tips * 100000000))
    echo "running $fasta ($tips tips, $steps steps) with seed $seed"

    "$delphy" \
        --v0-in-fasta "data/fasta/$fasta" \
        --v0-seed "$seed" \
        --v0-steps "$steps" \
        --v0-out-log-file "$out/$name.log" \
        --v0-out-trees-file "$out/$name.trees" \
        --v0-out-delphy-file "$out/$name.dphy" \
        ${DELPHY_FLAGS:-}
done
