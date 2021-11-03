# a-proof-time-series


This Github is a mini project based on the a-proof and a-proof-zonmw, both developed by CLTL.
The goal is to build a time series analysis model to predict rehabilitation behavior per domain and level for a new patient overtime, using only synthethic labels from the icf-classifier developed by in the a-prrof project.

The synthetic labels generated by the icf-classifier for 9 WHO-ICF domains are:

The domains are as follows:

| Domain | Dataset
| :---         | :---        
| Energy level  | (ENR) | 
| Attention functions	 | (ATT) | 
| Emotional functions	 | (STM) | 
| Respiration functions	 | (ADM) | 
| Exercise tolerance functions	 | (INS) | 
| Weight maintenance functions	 | (MBW) | 
| Walking	 | (FAC) | 
| Eating	 | (ETN) | 
| Work and employment	 | (BER) | 

Functioning Levels
They are further identified regarding their levels to clinical text:
- FAC and INS have a scale of 0-5, where 5 means there is no functioning problem.
- The rest of the domains have a scale of 0-4, where 4 means there is no functioning problem.

For more details on the ICF domains and levels please refer to aproof-icf-classifier.

## Repository overview:

This repository is organized as follows:
``` bash
/a-proof-time-series
├── README.md
├── figures
├── processed
│   ├── covid_data_with_levels_anonimized.csv
│   ├── covid_data_without_levels_anonimized.csv
│   ├── cross_id_dom.csv
│   ├── feature_engineering.csv
│   └── filtered_data.csv
├── processed 0-statistics_for_medical_team explained.ipynb
├── 1-data_exploration.ipynb
├── 2-trend_evaluation.ipynb
├── 3-feature_engineering.ipynb
└── 4-modeling.ipynb
```

This repository contains 5 notebooks with the data statistics, exploration, trend evaluation, feature engineering and modeling for the dataset described above.
It also contains figures and processed files that can help this project being reproduced.

The notebook 0-statistics can be run independently and should suffice for data statistics.
For modeling purposes please follow execution of notebooks 2,3 and 4 in order, or download the files from the 'processed' folder, which contains intermediate results.



## Related repositories
a-proof-icf-classifier: the final end-to-end machine learning pipeline for assigning the 9 WHO-ICF domains and their levels to clinical text. This is the final product of the experiments conducted in the current repo.

a-proof: the pilot phase preceding the current project. In the pilot, 4 WHO-ICF domains and their levels were annotated in about 5,000 clinical notes. Pre-trained BERTje vectors were used to encode the annotated sentences. SVM classifier was trained for the domains, and a regression model was trained for the levels.
Dutch medical language model: code for creating and evaluating the medical/clinical language model that is fine-tuned in the current repository. 

## Disclaimer
Please note patient IDS (named as MDN here) were anonimized since the file contains sensitive information over medical data.


