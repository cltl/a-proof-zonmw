a-proof-zonmw
=============

# Description
The goal of the A-PROOF/ZonMw project is to create classifiers that identify the functioning level of a patient from a free-text clinical note in Dutch. We focus on 9 [WHO-ICF](https://www.who.int/standards/classifications/international-classification-of-functioning-disability-and-health) domains, which were chosen due to their relevance to recovery from COVID-19:

ICF code | Domain
---|---
b1300 | Energy level
b140 | Attention functions
b152 | Emotional functions
b440 | Respiration functions
b455 | Exercise tolerance functions
b530 | Weight maintenance functions
d450 | Walking
d550 | Eating
d840-d859 | Work and employment

This repo contains the code and resources used in the course of the project. For the final machine learning pipeline that can be applied to new data and generate predictions, refer to [a-proof-icf-classifier](https://github.com/cltl/aproof-icf-classifier).

# Contents
1. [Requirements](#requirements)
2. [Repo structure](#repo-structure)
3. [Data](#data)
4. [Related repositories](#related-repositories)

# Requirements
The requirements are listed in the [environment.yml](environment.yml) file. It is recommended to create a virtual environment with conda (you need to have Anaconda or Miniconda installed):
```
$ conda env create -f environment.yml
$ conda activate zonmw
```

# Repo structure
The repo is organized as follows:
- `clf_domains`: scripts for training and evaluating a multi-label classification model that detects the 9 ICF domains. 
- `clf_levels`: scripts for training and evaluating a regression model that assigns a level of functioning per domain.
- `data_process`: scripts for various data processing tasks, incl. processing of raw data, data prep for annotation, processing annotations, etc.
- `nb_data_analysis`: notebooks to generate descriptive statistics (tables and figures) about the data.
- `nb_iaa`: notebooks for inter-annotator-agreement analysis.
- `nb_ml_evaluation`: notebooks for evaluation of the machine learning models.
- `resources`: files (json, xlsx) used for configuring the annotation environment and performing keyword searches in the data.
- `utils`: general helper functions used throughout the repo.

For details, please refer to the READMEs in the individual directories.

# Data
The data for the project consists of clinical notes from Electronic Health Records (EHRs) in Dutch. Due to privacy constraints, the data cannot be released.

# Related repositories
1. [a-proof-icf-classifier](https://github.com/cltl/aproof-icf-classifier): the final end-to-end machine learning pipeline for assigning the 9 WHO-ICF domains and their levels to clinical text. This is the final product of the experiments conducted in the current repo.
2. [a-proof](https://github.com/cltl/a-proof): the pilot phase preceding the current project. In the pilot, 4 WHO-ICF domains and their levels were annotated in about 5,000 clinical notes. Pre-trained [BERTje](https://github.com/wietsedv/bertje) vectors were used to encode the annotated sentences. SVM classifier was trained for the domains, and a regression model was trained for the levels.
3. [Dutch medical language model](https://github.com/cltl-students/verkijk_stella_rma_thesis_dutch_medical_langauge_model): code for creating and evaluating the medical/clinical language model that is fine-tuned in the current repository.