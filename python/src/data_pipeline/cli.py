"""Generates BEAST XMLs from FASTA files and the partial XMLs in `templates/`."""

import argparse
import sys
from pathlib import Path

from src.data_pipeline.build import (
    DEFAULT_CHAIN_LENGTH,
    DEFAULT_STORE_EVERY,
    list_models,
    make_dataset_id,
    render,
)
from src.data_pipeline.fasta import read_fasta, read_metadata_dates

REPO_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", nargs="+", help="FASTA files, with `name|date` headers unless --metadata is given")
    parser.add_argument("--metadata", help="table with the sampling dates (.xls/.xlsx, or delimited text: .tsv tab, else comma)")
    parser.add_argument("--id-column", default="Accession", help="metadata column matching the FASTA ids")
    parser.add_argument("--date-column", default="Collection_Date", help="metadata column with the dates")
    parser.add_argument("--id-field", type=int, help="0-based `|` field of the FASTA header holding the id (default: first word)")
    parser.add_argument("--skip-unmatched", action="store_true", help="drop FASTA records without a metadata date")
    parser.add_argument("--delimiter", help="metadata delimiter (default: by file extension)")
    parser.add_argument("--models", nargs="+", default=list_models(), choices=list_models(), help="templates to render")
    parser.add_argument("--id", help="dataset id used in the XML (default: the sanitised FASTA name)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "xmls", help="output directory")
    parser.add_argument("--chain-length", type=int, default=DEFAULT_CHAIN_LENGTH, help="MCMC chain length")
    parser.add_argument("--store-every", type=int, default=DEFAULT_STORE_EVERY, help="state store interval")
    parser.add_argument("--force", action="store_true", help="overwrite existing XMLs")
    args = parser.parse_args()

    if args.id and len(args.fasta) > 1:
        parser.error("--id can only be used with a single FASTA file")

    dates = None
    if args.metadata:
        dates = read_metadata_dates(args.metadata, args.id_column, args.date_column, args.delimiter)
        print(f"{args.metadata}: {len(dates)} dates", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fasta_path in args.fasta:
        taxa = read_fasta(fasta_path, dates, args.skip_unmatched, args.id_field)
        dataset_id = args.id or make_dataset_id(Path(fasta_path).stem)
        print(f"{fasta_path}: {len(taxa)} taxa, {len(taxa[0].sequence)} sites", file=sys.stderr)
        for model in args.models:
            output = args.out_dir / f"{model}_{dataset_id}.xml"
            if output.exists() and not args.force:
                print(f"skipping existing {output} (use --force)", file=sys.stderr)
                continue
            output.write_text(render(model, taxa, dataset_id, args.chain_length, args.store_every))
            print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
