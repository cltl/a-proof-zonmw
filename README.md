a-proof-zonmw
=============

# Description
The goal of the A-PROOF/ZonMw project is to create classifiers that identify the functioning level of a patient from a free-text clinical note in Dutch. We focus on 9 [WHO-ICF](https://www.who.int/standards/classifications/international-classification-of-functioning-disability-and-health) domains, which were chosen due to their relevance to recovery from COVID-19:

ICF code | Domain | name in repo
---|---|---
b1300 | Energy level | ENR
b140 | Attention functions | ATT
b152 | Emotional functions | STM
b440 | Respiration functions | ADM
b455 | Exercise tolerance functions | INS
b530 | Weight maintenance functions | MBW
d450 | Walking | FAC
d550 | Eating | ETN
d840-d859 | Work and employment | BER

This repo contains the code and resources used in the course of the project. For the final machine learning pipeline that can be applied to new data and generate predictions, refer to [a-proof-icf-classifier](https://github.com/cltl/aproof-icf-classifier).

# Contents
1. [Requirements](#requirements)
2. [Repo structure](#repo-structure)
3. [Configuring and calling paths](#configuring-and-calling-paths)
4. [Data](#data)
5. [Related repositories](#related-repositories)

# Requirements
The requirements are listed in the [environment.yml](environment.yml) file. It is recommended to create a virtual environment with conda (you need to have Anaconda or Miniconda installed):
```
$ conda env create -f environment.yml
$ conda activate zonmw
```

# Repo structure
The repo is organized as follows:
- `clf_domains`: scripts for training and evaluating a multi-label classification model that detects the 9 ICF domains.
- `clf_levels`: scripts for training and evaluating regression models that assign a level of functioning per domain.
- `data_process`: scripts for various data processing tasks, incl. processing of raw data, data prep for annotation, processing annotations, data prep for the machine learning pipeline etc.
- `ml_evaluation`: scripts and notebooks for evaluation of the machine learning models.
- `nb_data_analysis`: notebooks to generate descriptive statistics (tables and figures) about the data.
- `nb_iaa`: notebooks for inter-annotator-agreement analysis.
- `resources`: annotation gudelines, files used for configuring the annotation environment, files for keyword searches in the data.
- `utils`: general helper functions used throughout the repo.

For details, please refer to the READMEs in the individual directories. A report can be found in the doc folder.

# Configuring and calling paths
- All paths that are used in the code of this repo are listed in [config.ini](config.ini).
- From the [config.py](utils/config.py) module, the `PATHS` object can be imported. All paths can be accessed from the `PATHS` object by calling `getpath` and providing the key listed in [config.ini](config.ini). This returns the path as a [pathlib](https://docs.python.org/3/library/pathlib.html) `Path` object.

Example:
```
from utils.config import PATHS

datapath = PATHS.getpath('data_expr_sept')
filepath = datapath / 'example.csv'
```

# Data
The data for the project consists of clinical notes from Electronic Health Records (EHRs) in Dutch. Due to privacy constraints, the data cannot be released.

# Related repositories
1. [a-proof-icf-classifier](https://github.com/cltl/aproof-icf-classifier): the final end-to-end machine learning pipeline for assigning the 9 WHO-ICF domains and their levels to clinical text. This is the final product of the experiments conducted in the current repo.
2. [a-proof](https://github.com/cltl/a-proof): the pilot phase preceding the current project. In the pilot, 4 WHO-ICF domains and their levels were annotated in about 5,000 clinical notes. Pre-trained [BERTje](https://github.com/wietsedv/bertje) vectors were used to encode the annotated sentences. SVM classifier was trained for the domains, and a regression model was trained for the levels.
3. [Dutch medical language model](https://github.com/cltl-students/verkijk_stella_rma_thesis_dutch_medical_langauge_model): code for creating and evaluating the medical/clinical language model that is fine-tuned in the current repository.
