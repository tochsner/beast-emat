
## Data pipeline

`src/data_pipeline` turns FASTA files (headers `name|YYYY[-MM[-DD]]`) into BEAST XMLs. The partial XMLs in `src/data_pipeline/templates/` (`beauti`, `emat`, `bd_emat`) hold everything except the data. The alignment, date trait, type trait, dataset id, chain length and store interval are filled in at the `{{PLACEHOLDER}}`s. To add a model, add a template.

```
data/fasta/<dataset>.fasta  ->  data/xmls/<model>_<dataset>.xml
```

Run it for one FASTA file, or for all datasets with the script (it first downloads or recovers missing FASTA files):

```
PYTHONPATH=python uv run --project python python -m src.data_pipeline.cli data/fasta/zika.fasta --models emat --id zika
scripts/generate_xmls.sh [--force]
```

The FASTA downloaders are in `src/data_pipeline/download/`: `cog_uk_to_fasta.py` (SARS-CoV-2, COG-UK/ENA), `seqset_to_fasta.py` (Pathoplexus SeqSet, one per `data/seqsets/*-seqset.json`) and `xml_to_fasta.py` (recover the alignment of an existing XML).
