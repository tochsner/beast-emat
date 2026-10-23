"""Reading of FASTA files with `name|date` headers."""

import calendar
import csv
import re
from dataclasses import dataclass
from pathlib import Path

DATE_PATTERN = re.compile(r"^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$")


@dataclass(frozen=True)
class Taxon:
    """A sequence record, with the full FASTA header as its name."""

    name: str
    raw_date: str
    date: str
    sequence: str


def normalise_date(raw_date):
    """Returns the date as YYYY-MM-DD, filling a missing month or day with the first one."""
    match = DATE_PATTERN.match(raw_date)
    if not match:
        raise ValueError(f"unparseable date '{raw_date}'")
    year, month, day = int(match[1]), int(match[2] or 1), int(match[3] or 1)
    if not 1 <= month <= 12 or not 1 <= day <= calendar.monthrange(year, month)[1]:
        raise ValueError(f"invalid date '{raw_date}'")
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_header(header):
    """Returns the (name, raw date, normalised date) of a `name|date` header."""
    if "|" not in header:
        raise ValueError(f"header '{header}' has no '|date' suffix")
    raw_date = header.rsplit("|", 1)[1]
    try:
        return header, raw_date, normalise_date(raw_date)
    except ValueError as error:
        raise ValueError(f"header '{header}': {error}") from error


def read_fasta(path, dates=None, skip_unmatched=False, id_field=None):
    """Returns the taxa of a FASTA file, raising on duplicate names or bad dates.

    Without `dates`, the headers must have the form `name|date`. With `dates`, a mapping from
    sequence id to date (see `read_metadata_dates`), the id is the first word of the header and
    the taxon is named `id|date`. The id is the first word of the header, or the `id_field`-th
    (0-based) `|`-separated field if given. Records without a date in the metadata raise, or are dropped
    if `skip_unmatched` is set.
    """
    taxa = []
    seen = set()
    for header, sequence in iterate_records(Path(path)):
        if dates is None:
            name, raw_date, date = parse_header(header)
        else:
            sequence_id = extract_id(header, id_field)
            if skip_unmatched and not find_raw_date(sequence_id, dates):
                continue
            name, raw_date, date = look_up_date(sequence_id, dates)
        if name in seen:
            raise ValueError(f"duplicate taxon '{name}'")
        seen.add(name)
        taxa.append(Taxon(name, raw_date, date, sequence.upper()))
    if not taxa:
        raise ValueError(f"no sequences in {path}")
    return taxa


def read_metadata_dates(path, id_column, date_column, delimiter=None):
    """Returns a mapping from sequence id to raw date from a metadata table.

    Excel files (.xls, .xlsx) are read with pandas. Other files are delimited text, with a tab
    for `.tsv` files and a comma otherwise, unless a delimiter is given.
    """
    path = Path(path)
    if path.suffix in (".xls", ".xlsx"):
        rows = read_excel_rows(path)
    else:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter or ("\t" if path.suffix == ".tsv" else ","))
            rows = list(reader)
    dates = {}
    for row in rows:
        for column in (id_column, date_column):
            if column not in row:
                raise ValueError(f"column '{column}' not in {path}, found {list(row)}")
        if row[id_column] in dates:
            raise ValueError(f"duplicate id '{row[id_column]}' in {path}")
        dates[row[id_column]] = row[date_column]
    return dates


def read_excel_rows(path):
    """Returns the rows of the first sheet as dicts of strings, with empty cells as empty strings."""
    import pandas as pd

    table = pd.read_excel(path, dtype=str).fillna("")
    return table.to_dict("records")


def extract_id(header, id_field=None):
    """Returns the sequence id of a header.

    This is the `id_field`-th (0-based) `|`-separated field if given, else the first word, where
    words are separated by whitespace or `|`.
    """
    if id_field is not None:
        fields = header.split("|")
        if id_field >= len(fields):
            raise ValueError(f"header '{header}' has no field {id_field}")
        return fields[id_field].strip()
    return re.split(r"[\s|]+", header.strip(), maxsplit=1)[0]


def find_raw_date(sequence_id, dates):
    """Returns the raw date of an id, trying the id without version suffix (`.1`) as well."""
    return dates.get(sequence_id) or dates.get(sequence_id.rsplit(".", 1)[0])


def look_up_date(sequence_id, dates):
    """Returns the (name, raw date, normalised date) of a sequence id that is in the metadata."""
    raw_date = find_raw_date(sequence_id, dates)
    if not raw_date:
        raise ValueError(f"no date in the metadata for sequence '{sequence_id}'")
    try:
        return f"{sequence_id}|{raw_date}", raw_date, normalise_date(raw_date)
    except ValueError as error:
        raise ValueError(f"sequence '{sequence_id}': {error}") from error


def iterate_records(path):
    """Yields (header, sequence) tuples, one per record."""
    header = None
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)
