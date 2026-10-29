#!/usr/bin/env bash
# run.sh <variant> <dataset> <seed> <seconds>: runs xmls/<variant>_<dataset>.xml of the benchmark directory
# that holds this script, with a wall-time cap, in runs/<variant>_<dataset>-<seed>/
S=$(cd "$(dirname "$0")" && pwd)
v=$1; d=$2; seed=$3; secs=$4

dir=$S/runs/${v}_${d}-$seed; rm -rf "$dir"; mkdir -p "$dir/data/out"; cd "$dir"
start=$(date +%s)
java -Xmx2g -XX:+UseSerialGC -Dstate.file.name=$dir/state --module-path "$S/classes:$(cat $S/deps.txt)" -DBEAST_PACKAGE_PATH=$S/classes \
    -m beast.base/beast.base.minimal.BeastMain -overwrite -seed $seed "$S/xmls/${v}_${d}.xml" > stdout.txt 2>&1 &
pid=$!

# macOS has no timeout, so a background sleep kills the chain at the cap
( sleep "$secs"; kill "$pid" 2>/dev/null ) &
killer=$!
wait "$pid"; code=$?
kill "$killer" 2>/dev/null
echo "$v $d $seed start=$start end=$(date +%s) exit=$code" > timing.txt
