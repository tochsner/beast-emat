"""Rendering of partial XMLs (templates) with the data of a FASTA file."""

import re
from pathlib import Path
from xml.sax.saxutils import quoteattr

TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_CHAIN_LENGTH = 500_000_000
DEFAULT_STORE_EVERY = 5000


def list_models():
    """Returns the names of the available templates."""
    return sorted(path.stem for path in TEMPLATE_DIR.glob("*.xml"))


def make_dataset_id(stem):
    """Returns an XML-safe dataset id derived from a file stem."""
    return re.sub(r"[^A-Za-z0-9]+", "", stem) or "dataset"


def render(model, taxa, dataset_id, chain_length=DEFAULT_CHAIN_LENGTH, store_every=DEFAULT_STORE_EVERY):
    """Returns the XML of the given model template filled with the taxa."""
    template = (TEMPLATE_DIR / f"{model}.xml").read_text()
    values = {
        "ALIGNMENT": render_alignment(taxa, dataset_id),
        "DATE_TRAIT": render_date_trait(taxa),
        "TYPE_TRAIT": ",".join(f"{taxon.name}=NOT_SET" for taxon in taxa),
        "CHAIN_LENGTH": str(chain_length),
        "STORE_EVERY": str(store_every),
        "ID": dataset_id,
    }

    # one pass, so that sequence data containing a placeholder cannot be substituted again
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda match: values[match[1]], template)


def render_alignment(taxa, dataset_id):
    """Returns the `<data>` element holding one `<sequence>` per taxon."""
    lines = [f'<data id="{dataset_id}" spec="Alignment" name="alignment">']
    for taxon in taxa:
        name = quoteattr(taxon.name)
        lines.append(
            f'        <sequence id={quoteattr("seq_" + taxon.name)} spec="Sequence" taxon={name} '
            f'totalcount="4" value="{taxon.sequence}"/>'
        )
    lines.append("    </data>")
    return "\n".join(lines)


def render_date_trait(taxa):
    """Returns the comma-separated `name=date` entries of the date trait."""
    return ",\n".join(f"{taxon.name}={taxon.date}" for taxon in taxa)
