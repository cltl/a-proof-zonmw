Prepare a batch for annotation
=============================
# Description
The main script of this repo is [prep_batch_for_annotation.py](prep_batch_for_annotation.py); the rest of the scripts contain helper functions.

The steps performed by the script:
1. Select notes for an annotation batch, based on the desired parameters. See [below](#customizing-the-batch) for more details about how to customize the parameters.
2. Convert the notes to CoNLL format and save them in folders per annotator.
3. Save an overview of the batch as a pickled DataFrame.

# Customizing the batch
To customize a batch, add a dictionary to [config.batch_prep.json](config.batch_prep.json), where the key is the name of the batch and the value is a dictionary defining the following parameters:
```
annotators: list
    list of the annotators
sources_other: list
    list of the names of non-covid datasets
source_covid: str
    name of the covid dataset
n_files: int
    number of notes per annotator
pct_covid: float
    desired fraction of covid data in the batch
pct_kwd: float
    desired fraction of notes containing keywords in the batch
domains: list
    all domains
matched_domains: list
    only notes that contain keyword matches from these domains are selected
min_matched_domains: int
    only notes that contain keyword matches from at least this number of domains are selected
n_iaa: int
    number of files that should be the same for all annotators (for IAA purposes)
iaa_sources: list
    list of the names of datasets from which IAA notes are selected
```
In addition, to the parameters defined in [config.batch_prep.json](config.batch_prep.json), you can customize the following as arguments of the [prep_batch_for_annotation.py](prep_batch_for_annotation.py) script:
```
kwdversion:
    the version of the keywords file to use
note_types:
    the type of notes to select (e.g. Voortgangsverslag)
```
