#!/usr/bin/env python3
"""Download the GenBank sequences listed in a metadata table as a FASTA file.

The table is tab-separated with an `accession` and a `date` column, such as the
Nextstrain community metadata in `data/genbank`. The sequences are fetched from
NCBI Nucleotide with the E-utilities, see
https://www.ncbi.nlm.nih.gov/books/NBK25499/.

The FASTA headers have the form `accession|YYYY-MM-DD`, with the date taken from
the table. Dates given as a range, such as `2022-12-31 (2022-07-26 - 2022-12-31)`,
use their first part. Sequences without a valid date are skipped.

The sequences are aligned with MAFFT, see https://mafft.cbrc.jp/alignment/software/,
which must be on the PATH. With `--reference`, each sequence is aligned to the reference
genome, otherwise the sequences are aligned to each other. The output is then an aligned
FASTA file of equal-length sequences with gaps. Pass `--no-align` to keep the unaligned
sequences.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from src.data_pipeline.fasta import normalise_date

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def base_accession(accession_version):
    """Returns an accession without its version suffix, e.g. `MN908947` for `MN908947.3`."""
    return accession_version.split(".")[0]


def read_dates(path):
    """Returns a mapping from accession to its raw date, keeping the first row of each accession."""
    dates = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            accession = base_accession(row["accession"].strip())
            if accession and accession not in dates:
                dates[accession] = row["date"].split(" ")[0]
    return dates


def request_with_retries(request, num_retries, delay):
    """Returns the decoded response body, retrying on transient server errors."""
    for attempt in range(num_retries + 1):
        try:
            with urllib.request.urlopen(request) as response:
                return response.read().decode()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt == num_retries:
                raise
            print(f"warning: retrying after {error}", file=sys.stderr)
            time.sleep(delay * (attempt + 1))


def fetch_fasta(accessions, email, num_retries=3, delay=2):
    """Returns the FASTA text of the given accessions from NCBI Nucleotide."""
    params = {"db": "nuccore", "id": ",".join(accessions), "rettype": "fasta", "retmode": "text", "tool": "beast-emat"}
    if email:
        params["email"] = email
    request = urllib.request.Request(EFETCH_URL, data=urllib.parse.urlencode(params).encode(), method="POST")
    return request_with_retries(request, num_retries, delay)


def parse_fasta(text):
    """Returns a list of (header, sequence) tuples."""
    records = []
    for entry in text.split(">")[1:]:
        header, _, sequence = entry.partition("\n")
        records.append((header.strip(), sequence.replace("\n", "")))
    return records


def run_mafft(options, inputs, num_threads):
    """Returns the aligned FASTA text that MAFFT writes for the given options and input files."""
    result = subprocess.run(
        ["mafft", *options, "--thread", str(num_threads), *inputs],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def add_to_reference(path, reference, num_threads):
    """Returns the sequences of a FASTA file aligned to a reference, without the reference itself."""
    with open(reference) as f:
        reference_name = parse_fasta(f.read())[0][0].split()[0]
    aligned = parse_fasta(run_mafft(["--6merpair", "--addfragments", str(path)], [str(reference)], num_threads))
    sequences = [(header, sequence) for header, sequence in aligned if header.split()[0] != reference_name]
    columns = [i for i in range(len(sequences[0][1])) if any(sequence[i] != "-" for _, sequence in sequences)]
    return "".join(f">{header}\n{''.join(sequence[i] for i in columns)}\n" for header, sequence in sequences)


def align_fasta(path, reference, num_threads):
    """Aligns the sequences of a FASTA file with MAFFT, replacing the file with the alignment."""
    if reference:
        aligned = add_to_reference(path, reference, num_threads)
    else:
        aligned = run_mafft(["--parttree", "--retree", "2"], [str(path)], num_threads)
    with open(path, "w") as f:
        f.write(aligned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", help="tab-separated metadata table with accession and date columns")
    parser.add_argument("output", help="output FASTA file")
    parser.add_argument("--email", help="contact address sent to NCBI")
    parser.add_argument("--batch-size", type=int, default=200, help="accessions per request")
    parser.add_argument(
        "--delay", type=float, default=0.4, help="seconds between requests (NCBI allows 3 per second without an API key)"
    )
    parser.add_argument("--no-align", dest="align", action="store_false", help="keep the sequences unaligned")
    parser.add_argument("--reference", help="reference genome FASTA to align the sequences to")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="MAFFT threads used for the alignment")
    args = parser.parse_args()

    dates = read_dates(args.metadata)
    accessions = list(dates)
    print(f"{len(accessions)} accessions in {args.metadata}", file=sys.stderr)

    sequences = {}
    for start in range(0, len(accessions), args.batch_size):
        batch = accessions[start : start + args.batch_size]
        text = fetch_fasta(batch, args.email)
        for header, sequence in parse_fasta(text):
            sequences[base_accession(header.split()[0])] = sequence
        print(f"downloaded {len(sequences)}/{len(accessions)}", file=sys.stderr)
        time.sleep(args.delay)

    num_missing = 0
    num_undated = 0
    with open(args.output, "w") as f:
        for accession, raw_date in dates.items():
            if accession not in sequences:
                num_missing += 1
                continue
            try:
                date = normalise_date(raw_date)
            except ValueError:
                num_undated += 1
                continue
            f.write(f">{accession}|{date}\n{sequences[accession]}\n")

    print(f"skipped {num_missing} accessions not found in GenBank", file=sys.stderr)
    print(f"skipped {num_undated} sequences without a valid date", file=sys.stderr)

    if args.align:
        print(f"aligning sequences in {args.output}", file=sys.stderr)
        align_fasta(args.output, args.reference, args.threads)


if __name__ == "__main__":
    main()
