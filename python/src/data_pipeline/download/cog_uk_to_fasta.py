#!/usr/bin/env python3
"""Download the sequences analysed in the COG-UK lineage dynamics study as a FASTA file.

The study https://github.com/COG-UK/UK-lineage-dynamics-analysis analysed 50,887
SARS-CoV-2 genomes, but cannot redistribute them because of the GISAID terms of
use. Its `data/phylogenetic/metadata.csv` lists the sequence names and sampling
dates only.

The 26,181 UK genomes among them were also deposited in the ENA by the COG-UK
consortium, where they are public. This script resolves those sequence names to
ENA accessions and downloads the consensus genomes, see
https://www.ebi.ac.uk/ena/portal/api. The non-UK genomes are only available from
GISAID, which needs an account, and are therefore skipped.

The ENA holds unaligned consensus genomes. With `--align` they are mapped onto
the Wuhan-Hu-1 reference (MN908947.3) with minimap2, via its `mappy` bindings,
so that every record has the reference length of 29,903 sites. This needs
`mappy`, which is a dependency of the `python` project in this repository.

The FASTA headers have the form `sequence_name|YYYY-MM-DD`, with the sampling
date taken from the study metadata.
"""

import argparse
import csv
import io
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

METADATA_URL = (
    "https://raw.githubusercontent.com/COG-UK/UK-lineage-dynamics-analysis"
    "/main/data/phylogenetic/metadata.csv"
)
PORTAL_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
BROWSER_URL = "https://www.ebi.ac.uk/ena/browser/api/fasta"
REFERENCE_ACCESSION = "MN908947.3"


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


def read_metadata(path):
    """Returns the study metadata rows, downloading them if no local path is given."""
    if path:
        with open(path) as f:
            return list(csv.DictReader(f))
    text = request_with_retries(urllib.request.Request(METADATA_URL), 3, 2)
    return list(csv.DictReader(io.StringIO(text)))


def to_alias(sequence_name):
    """Returns the ENA sample alias of a study sequence name, e.g. `England/SHEF-C0181/2020`."""
    return "COG-UK/" + sequence_name.split("/")[1]


def query_portal(result, field, values, fields, num_retries, delay):
    """Returns the rows of an ENA portal search matching any of the given field values."""
    query = " OR ".join(f'{field}="{value}"' for value in values)
    body = urllib.parse.urlencode(
        {"result": result, "query": query, "fields": fields, "format": "tsv", "limit": 0}
    ).encode()
    request = urllib.request.Request(PORTAL_URL, data=body, method="POST")
    text = request_with_retries(request, num_retries, delay)
    if not text.strip():
        return []
    lines = text.strip().split("\n")
    return [line.split("\t") for line in lines[1:]]


def resolve_accessions(aliases, batch_size, num_retries, delay):
    """Returns a mapping from sample alias to ENA sequence accession."""

    # aliases resolve to sample accessions, which in turn carry the assembled genomes
    sample_to_alias = {}
    for start in range(0, len(aliases), batch_size):
        batch = aliases[start : start + batch_size]
        for sample_accession, alias in query_portal(
            "sample", "sample_alias", batch, "sample_accession,sample_alias", num_retries, delay
        ):
            sample_to_alias[sample_accession] = alias
        print(f"resolved {len(sample_to_alias)} samples ({start + len(batch)}/{len(aliases)})", file=sys.stderr)

    sample_accessions = list(sample_to_alias)
    alias_to_sequence = {}
    for start in range(0, len(sample_accessions), batch_size):
        batch = sample_accessions[start : start + batch_size]
        for accession, sample_accession in query_portal(
            "sequence", "sample_accession", batch, "accession,sample_accession", num_retries, delay
        ):
            alias = sample_to_alias[sample_accession]

            # a sample can carry several assemblies, keep the first accession for reproducibility
            if alias not in alias_to_sequence or accession < alias_to_sequence[alias]:
                alias_to_sequence[alias] = accession
        print(
            f"resolved {len(alias_to_sequence)} sequences ({start + len(batch)}/{len(sample_accessions)})",
            file=sys.stderr,
        )

    return alias_to_sequence


def parse_fasta(text):
    """Returns a list of (header, sequence) tuples."""
    records = []
    for entry in text.split(">")[1:]:
        header, _, sequence = entry.partition("\n")
        records.append((header.strip(), sequence.replace("\n", "")))
    return records


def fetch_sequences(accessions, batch_size, num_retries, delay):
    """Returns a mapping from ENA sequence accession to genome."""
    sequences = {}
    for start in range(0, len(accessions), batch_size):
        batch = accessions[start : start + batch_size]
        url = f"{BROWSER_URL}/{','.join(batch)}"
        text = request_with_retries(urllib.request.Request(url), num_retries, delay)
        for header, sequence in parse_fasta(text):

            # the headers have the form `ENA|OX567185|OX567185.1 <description>`
            accession = header.split("|")[1]
            sequences[accession] = sequence
        print(f"downloaded {len(sequences)}/{len(accessions)} sequences", file=sys.stderr)
    return sequences


def fetch_reference(path, num_retries, delay):
    """Returns the reference genome, downloading it if no local path is given."""
    if path:
        with open(path) as f:
            return parse_fasta(f.read())[0][1]
    url = f"{BROWSER_URL}/{REFERENCE_ACCESSION}"
    text = request_with_retries(urllib.request.Request(url), num_retries, delay)
    return parse_fasta(text)[0][1]


def make_aligner(reference):
    """Returns a minimap2 aligner against the reference genome."""
    try:
        import mappy
    except ImportError:
        sys.exit(
            "error: --align needs the mappy package, install it with `uv add mappy` in python/"
        )
    return mappy.Aligner(seq=reference, preset="asm20")


def align_to_reference(aligner, sequence, reference_length):
    """Returns a genome projected onto reference coordinates, or None if it does not align.

    Sites deleted with respect to the reference become gaps and sites that are not
    covered by an alignment become `N`. Insertions with respect to the reference are
    dropped, which keeps every genome at the reference length.
    """
    import mappy

    sites = ["N"] * reference_length
    is_aligned = False
    for hit in aligner.map(sequence):
        if not hit.is_primary:
            continue
        is_aligned = True

        # for a reverse hit the query coordinates refer to the reverse complement
        query = sequence if hit.strand > 0 else mappy.revcomp(sequence)
        reference_position, query_position = hit.r_st, hit.q_st
        for length, operation in hit.cigar:

            # the operations are minimap2's, i.e. 0 = match, 1 = insertion, 2 = deletion
            if operation == 0:
                sites[reference_position : reference_position + length] = query[
                    query_position : query_position + length
                ]
                reference_position += length
                query_position += length
            elif operation == 1:
                query_position += length
            elif operation == 2:
                sites[reference_position : reference_position + length] = "-" * length
                reference_position += length

    return "".join(sites) if is_aligned else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="output FASTA file")
    parser.add_argument("--num-sequences", type=int, default=10_000, help="number of sequences to subsample")
    parser.add_argument("--metadata", help="local copy of the study metadata.csv (default: download)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for subsampling (default: 0)")
    parser.add_argument("--max-ambiguous", type=float, default=0.05, help="maximum fraction of N per sequence")
    parser.add_argument("--align", action="store_true", help="align the genomes to the reference")
    parser.add_argument("--reference", help="local copy of the reference genome (default: download)")
    parser.add_argument("--batch-size", type=int, default=500, help="accessions per request")
    parser.add_argument("--retries", type=int, default=3, help="retries per failed request")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="seconds to wait before a retry")
    args = parser.parse_args()

    rows = read_metadata(args.metadata)
    print(f"{len(rows)} sequences in the study metadata", file=sys.stderr)

    # only the UK genomes are public, the rest is GISAID-only
    dates = {}
    for row in rows:
        if row["country"] == "UK" and row["sample_date"]:
            dates[to_alias(row["sequence_name"])] = (row["sequence_name"], row["sample_date"])
    print(f"{len(dates)} UK sequences to resolve", file=sys.stderr)

    alias_to_sequence = resolve_accessions(sorted(dates), args.batch_size, args.retries, args.retry_delay)
    print(f"{len(alias_to_sequence)} of {len(dates)} UK sequences are available in the ENA", file=sys.stderr)

    aligner = None
    reference_length = 0
    if args.align:
        reference = fetch_reference(args.reference, args.retries, args.retry_delay)
        reference_length = len(reference)
        aligner = make_aligner(reference)
        print(f"aligning to a reference of {reference_length} sites", file=sys.stderr)

    rng = random.Random(args.seed)
    aliases = sorted(alias_to_sequence)
    rng.shuffle(aliases)

    # sequences are dropped below, so resolve a surplus and truncate once the target is met
    num_written = 0
    num_ambiguous = 0
    num_unaligned = 0
    with open(args.output, "w") as f:
        for start in range(0, len(aliases), args.batch_size):
            if num_written >= args.num_sequences:
                break
            batch = aliases[start : start + args.batch_size]
            accessions = [alias_to_sequence[alias] for alias in batch]
            sequences = fetch_sequences(accessions, args.batch_size, args.retries, args.retry_delay)
            for alias in batch:
                if num_written >= args.num_sequences:
                    break
                sequence = sequences.get(alias_to_sequence[alias])
                if not sequence:
                    continue
                if aligner:
                    sequence = align_to_reference(aligner, sequence, reference_length)
                    if not sequence:
                        num_unaligned += 1
                        continue
                if sequence.count("N") / len(sequence) > args.max_ambiguous:
                    num_ambiguous += 1
                    continue
                sequence_name, date = dates[alias]
                f.write(f">{sequence_name}|{date}\n{sequence}\n")
                num_written += 1
            print(f"wrote {num_written}/{args.num_sequences} sequences", file=sys.stderr)

    print(f"skipped {num_ambiguous} sequences with too many ambiguous sites", file=sys.stderr)
    if num_unaligned:
        print(f"skipped {num_unaligned} sequences that did not align", file=sys.stderr)
    if num_written < args.num_sequences:
        print(f"warning: only {num_written} sequences available", file=sys.stderr)
    print(f"wrote {num_written} sequences to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
