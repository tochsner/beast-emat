#!/usr/bin/env python3
"""Recover a FASTA file from the alignment of a BEAST XML.

The FASTA headers are the taxon names of the XML, which have the form
`name|date` in the XMLs of this repository.
"""

import argparse
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="BEAST XML with an Alignment")
    parser.add_argument("output", help="output FASTA file")
    args = parser.parse_args()

    num_written = 0
    with open(args.output, "w") as f:
        for _, element in ET.iterparse(args.xml):
            if element.tag == "sequence":
                f.write(f">{element.get('taxon')}\n{element.get('value')}\n")
                num_written += 1
                element.clear()
    print(f"wrote {num_written} sequences to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
