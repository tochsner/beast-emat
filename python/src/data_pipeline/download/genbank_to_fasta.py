#!/usr/bin/env python3
"""Download the GenBank sequences listed in a metadata table as a FASTA file.

The table is tab-separated with an `accession` and a `date` column, such as the
Nextstrain community metadata in `data/genbank`. The sequences are fetched from
NCBI Nucleotide with the E-utilities, see
https://www.ncbi.nlm.nih.gov/books/NBK25499/.

The FASTA headers have the form `accession|YYYY-MM-DD`, with the date taken from
the table. Dates given as a range, such as `2022-12-31 (2022-07-26 - 2022-12-31)`,
use their first part. Sequences without a valid date are skipped.
"""

import argparse
import csv
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", help="tab-separated metadata table with accession and date columns")
    parser.add_argument("output", help="output FASTA file")
    parser.add_argument("--email", help="contact address sent to NCBI")
    parser.add_argument("--batch-size", type=int, default=200, help="accessions per request")
    parser.add_argument(
        "--delay", type=float, default=0.4, help="seconds between requests (NCBI allows 3 per second without an API key)"
    )
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


if __name__ == "__main__":
    main()
