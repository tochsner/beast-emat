#!/usr/bin/env bash
# Generates the BEAST XMLs (beauti, emat, bd_emat) of all current datasets into data/xmls.
# The FASTA files (and metadata) are expected in data/fasta. Only the SARS-CoV-2 (COG-UK) and West Nile
# virus FASTA files are downloaded if missing.
# Usage: scripts/generate_xmls.sh [--force] [--chain-length N] [--store-every N]
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=python
pipeline_args=("$@")

run_python() {
    uv run --project python python -m "$@"
}

# fasta_from_cog <fasta name> <number of sequences>: downloads, aligns and subsamples SARS-CoV-2
fasta_from_cog() {
    local fasta="data/fasta/$1.fasta"
    if [[ ! -f "$fasta" ]]; then
        run_python src.data_pipeline.download.cog_uk_to_fasta "$fasta" --num-sequences "$2" --align
    fi
}

# fasta_from_seqset: downloads the West Nile virus sequences of the Pathoplexus SeqSet
fasta_from_seqset() {
    local fasta="data/fasta/wnv-pathoplexus.fasta"
    if [[ ! -f "$fasta" ]]; then
        run_python src.data_pipeline.download.seqset_to_fasta data/seqsets/wnv-pathoplexus-seqset.json "$fasta"
    fi
}

# generate <fasta name> <dataset id> [extra cli args]: renders all templates for a FASTA file
generate() {
    local fasta="data/fasta/$1.fasta" id="$2"
    shift 2
    run_python src.data_pipeline.cli "$fasta" --id "$id" "$@" ${pipeline_args[@]+"${pipeline_args[@]}"}
}

fasta_from_cog cog-uk-sars-cov-2-aligned-5000 5000
fasta_from_cog cog-uk-sars-cov-2-aligned 10000
fasta_from_seqset

generate zika zika
generate ebola ebola
generate cog-uk-sars-cov-2-aligned-5000 sars5000
generate cog-uk-sars-cov-2-aligned sars10000
generate wnv-pathoplexus wnv

# the NCBI SARS-CoV-2 FASTA has no dates in its headers, they come from the metadata CSV
ncbi=NCBI_SC2_countries_nextclade.aligned.clean.subsampled
generate "$ncbi" ncbisc2 --metadata "data/fasta/$ncbi.csv" --id-column Accession --date-column Collection_Date

# the GISAID FASTA files have their dates in the metadata exports. Each has a reference sequence without
# metadata, which is dropped. Influenza headers start with the isolate id, hCoV-19 and RSV headers have it in field 1.
generate gisaid_epiflu_H1_sequence_2024-07-01_to_2026-02-25_with_ref_aligned2_cleaned h1n1ha \
    --metadata "data/fasta/gisaid_epiflu_H1_isolates _2024-07-01_to_2026-02-25.xls" \
    --id-column Isolate_Id --date-column Collection_Date --skip-unmatched
generate gisaid_epiflu_H3_sequence_2024-07-01_to_2026-02-25_with_ref_aligned h3n2ha \
    --metadata "data/fasta/gisaid_epiflu_H3_isolates _2024-07-01_to_2026-02-25.xls" \
    --id-column Isolate_Id --date-column Collection_Date --skip-unmatched
generate gisaid_hcov-19_2024-07-01_to_2026_02_25_with_ref_aligned gisaidsc2 \
    --metadata data/fasta/gisaid_hcov-19_2024-07-01_to_2026_02_25.tsv \
    --id-column "Accession ID" --date-column "Collection date" --id-field 1 --skip-unmatched
generate gisaid_RSVA_2024-07-01_to_2026-02-25_with_ref_aligned rsva \
    --metadata data/fasta/gisaid_RSVA_2024-07-01_to_2026-02-25.tsv \
    --id-column "Accession ID" --date-column "Collection date" --id-field 1 --skip-unmatched
generate gisaid_RSVB_2024-07-01_to_2026-02-25_with_ref_aligned rsvb \
    --metadata data/fasta/gisaid_RSVB_2024-07-01_to_2026-02-25.tsv \
    --id-column "Accession ID" --date-column "Collection date" --id-field 1 --skip-unmatched
