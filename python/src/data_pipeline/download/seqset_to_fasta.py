#!/usr/bin/env python3
"""Download the sequences of a Pathoplexus SeqSet export as a FASTA file.

The SeqSet export (JSON) lists accession versions. The sequences are fetched
from the Pathoplexus LAPIS API, see
https://pathoplexus.org/docs/how-to/search-download-seqs-api.

The FASTA headers have the form `accessionVersion|YYYY-MM-DD`. Sequences
without a collection date are skipped. For partial dates (year or month only)
and date ranges, a day is drawn uniformly at random from the covered days.
"""

import argparse
import calendar
import datetime
import json
import random
import sys
import urllib.request

LAPIS_URL = "https://lapis.pathoplexus.org"


def read_accession_versions(seqset_path):
    """Returns the accession versions listed in a SeqSet export file."""
    with open(seqset_path) as f:
        seqset = json.load(f)
    return [seq["accession"] for seq in seqset["sequences"]]


def fetch_fasta(organism, accession_versions, aligned, header_template):
    """Returns the FASTA text for the given accession versions."""
    endpoint = "alignedNucleotideSequences" if aligned else "unalignedNucleotideSequences"
    url = f"{LAPIS_URL}/{organism}/sample/{endpoint}"
    body = json.dumps(
        {"accessionVersion": accession_versions, "fastaHeaderTemplate": header_template}
    ).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode()


def parse_date_bounds(date):
    """Returns the first and last possible day of a (partial) date or date range."""
    if "/" in date:
        start, end = date.split("/")
        return parse_date_bounds(start)[0], parse_date_bounds(end)[1]
    parts = [int(part) for part in date.split("-")]
    if len(parts) == 1:
        return datetime.date(parts[0], 1, 1), datetime.date(parts[0], 12, 31)
    if len(parts) == 2:
        last_day = calendar.monthrange(parts[0], parts[1])[1]
        return datetime.date(parts[0], parts[1], 1), datetime.date(parts[0], parts[1], last_day)
    day = datetime.date(*parts)
    return day, day


def sample_full_date(date, rng):
    """Returns a full date drawn uniformly from the days covered by a (partial) date or date range."""
    first, last = parse_date_bounds(date)
    return first + datetime.timedelta(days=rng.randint(0, (last - first).days))


def parse_fasta(text):
    """Returns a list of (header, sequence) tuples."""
    records = []
    for entry in text.split(">")[1:]:
        header, _, sequence = entry.partition("\n")
        records.append((header.strip(), sequence.replace("\n", "")))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seqset", help="SeqSet export file (JSON)")
    parser.add_argument("output", help="output FASTA file")
    parser.add_argument("--organism", default="west-nile", help="LAPIS organism (default: west-nile)")
    parser.add_argument(
        "--unaligned", action="store_true", help="download unaligned instead of aligned sequences"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for imputing days (default: 0)")
    parser.add_argument("--batch-size", type=int, default=1000, help="accessions per request")
    args = parser.parse_args()

    accession_versions = read_accession_versions(args.seqset)
    print(f"{len(accession_versions)} accessions in SeqSet", file=sys.stderr)

    rng = random.Random(args.seed)
    template = "{accessionVersion}\t{sampleCollectionDate}"
    records = {}
    for start in range(0, len(accession_versions), args.batch_size):
        batch = accession_versions[start : start + args.batch_size]
        text = fetch_fasta(args.organism, batch, not args.unaligned, template)
        for header, sequence in parse_fasta(text):
            accession_version, _, date = header.partition("\t")
            records[accession_version] = (date, sequence)
        print(f"downloaded {len(records)}/{len(accession_versions)}", file=sys.stderr)

    missing = [acc for acc in accession_versions if acc not in records]
    if missing:
        print(f"warning: {len(missing)} accessions not found: {missing[:10]}", file=sys.stderr)

    num_undated = 0
    num_imputed = 0
    with open(args.output, "w") as f:
        for accession_version in accession_versions:
            if accession_version not in records:
                continue
            date, sequence = records[accession_version]
            if not date:
                num_undated += 1
                continue
            full_date = sample_full_date(date, rng).isoformat()
            if full_date != date:
                num_imputed += 1
            f.write(f">{accession_version}|{full_date}\n{sequence}\n")

    print(f"skipped {num_undated} sequences without a date", file=sys.stderr)
    print(f"imputed the day of {num_imputed} sequences with a partial date", file=sys.stderr)


if __name__ == "__main__":
    main()
