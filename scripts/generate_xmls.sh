#!/usr/bin/env bash
# Generates the BEAST XMLs (beauti, emat, bd_emat) of all current datasets into data/xmls.
# The FASTA files (and metadata) are expected in data/fasta. Only the SARS-CoV-2 (COG-UK), West Nile
# virus and GenBank FASTA files are downloaded if missing.
# Usage: scripts/generate_xmls.sh [--force] [--chain-length N] [--store-every N]

# setup

# re-run under bash when started by another shell, e.g. sh, which may not support pipefail
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=python
pipeline_args=("$@")

run_python() {
    uv run --project python python -m "$@"
}

# download functions

# fasta_from_cog <fasta name> <number of sequences>: downloads, aligns and subsamples SARS-CoV-2
fasta_from_cog() {
    local fasta="data/fasta/$1.fasta"
    if [[ ! -f "$fasta" ]]; then
        run_python src.data_pipeline.download.cog_uk_to_fasta "$fasta" --num-sequences "$2" --align
    fi
}

# seqset_organism <seqset stem>: the LAPIS organism of a Pathoplexus SeqSet, the stem itself unless listed here
seqset_organism() {
    case "$1" in
        wnv-pathoplexus) echo west-nile ;;
        *) echo "$1" ;;
    esac
}

# seqset_id <seqset stem>: the dataset id of a Pathoplexus SeqSet, the stem without separators unless listed here
seqset_id() {
    case "$1" in
        wnv-pathoplexus) echo wnv ;;
        *) echo "${1//[^A-Za-z0-9]/}" ;;
    esac
}

# fasta_from_seqset <seqset file>: downloads the sequences of a Pathoplexus SeqSet
fasta_from_seqset() {
    local stem fasta
    stem=$(basename "$1" -seqset.json)
    fasta="data/fasta/$stem.fasta"
    if [[ ! -f "$fasta" ]]; then
        run_python src.data_pipeline.download.seqset_to_fasta "$1" "$fasta" --organism "$(seqset_organism "$stem")"
    fi
}

# fasta_from_genbank <metadata file> <fasta name> [reference FASTA]: downloads and aligns the GenBank sequences listed in a metadata table
fasta_from_genbank() {
    local fasta="data/fasta/$2.fasta"
    if [[ ! -f "$fasta" ]]; then
        if [[ -n "${3:-}" ]]; then
            run_python src.data_pipeline.download.genbank_to_fasta "data/genbank/$1" "$fasta" --reference "$3"
        else
            run_python src.data_pipeline.download.genbank_to_fasta "data/genbank/$1" "$fasta"
        fi
    fi
}

# generation function

# generate <fasta name> <dataset id> [extra cli args]: renders all templates for a FASTA file
generate() {
    local fasta="data/fasta/$1.fasta" id="$2"
    shift 2
    run_python src.data_pipeline.cli "$fasta" --id "$id" "$@" ${pipeline_args[@]+"${pipeline_args[@]}"}
}

# downloads

fasta_from_cog cog-uk-sars-cov-2-aligned-5000 5000
fasta_from_cog cog-uk-sars-cov-2-aligned-10000 10000
fasta_from_cog cog-uk-sars-cov-2-aligned-20000 20000
for seqset in data/seqsets/*-seqset.json; do
    fasta_from_seqset "$seqset"
done
fasta_from_genbank nextstrain_community_cevo-public_ReVSeq-project_adenovirus_b1_whole_metadata.tsv genbank-adenovirus-b1 "data/reference/adenovirus_b1_ NC_011203.1.fasta"
fasta_from_genbank nextstrain_community_cevo-public_ReVSeq-project_hmpv_all_genome_metadata.tsv genbank-hmpv "data/reference/hmpv_ NC_039199.1.fasta"

# xml generation

# datasets whose FASTA headers already hold the dates

generate zika zika
generate ebola ebola
generate sars-757 sars-757
generate cog-uk-sars-cov-2-aligned-5000 sars5000
generate cog-uk-sars-cov-2-aligned-10000 sars10000
generate cog-uk-sars-cov-2-aligned-20000 sars20000
for seqset in data/seqsets/*-seqset.json; do
    stem=$(basename "$seqset" -seqset.json)
    generate "$stem" "$(seqset_id "$stem")"
done
generate genbank-adenovirus-b1 adenovirusb1
generate genbank-hmpv hmpv

# datasets whose dates come from a metadata table

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
