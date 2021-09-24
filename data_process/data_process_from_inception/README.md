Process annotations
==================
# Description
The workflow for processing annotations:
1. [process_annotated.py](process_annotated.py): processes a batch of annotated tsv files (INCEpTION output) into one DataFrame, and saves it in pkl format.
2. [parse_annotations.py](parse_annotations.py): processes the output of the previous step (pickled DataFrame): the labels assigned by the annotators are parsed into separate columns (e.g. `ENR`, `ENR_lvl`, `background`, etc.), and the resulting parsed DataFrame is saved to pkl format.

# Filename conventions
## Default (a-proof-zonmw)
By default, the script assumes the file-naming conventions of a-proof-zonmw:
```
'institution--year--MDN--NotitieID--batch.conll'
```
Example:
```
'vumc--2020--1234567--123456789--batch3.conll'
```
If you need to process legacy annotations from the a-proof pilot project, you can do it by passing the `--legacy_parser` (and `--path_to_raw`) argument(s), as explained below.

## Legacy stella
The `legacy_stella` file-naming convention is:
```
'institution--idx--MDN--NotitieID--NotitieCSN--Notitiedatum--q--search.conll'
```
Example:
```
'AMC--123--1234567--123456789--123456789--2020-05-18--q1_q2_q3--Search1.conll'
```
To process this type of files, pass:
```
$ python process_annotated.py --legacy_parser legacy_stella
```

## Legacy marten
The `legacy_marten` file-naming convention is:
```
'rawfile---idx+1.conll'
```
Example:
```
'notities_2017_deel2_cleaned.csv---2276.conll'
```
To process this type of files, pass:
```
$ python process_annotated.py --legacy_parser legacy_marten --path_to_raw ../data/raw/notities_2017_deel2_cleaned.csv
```
