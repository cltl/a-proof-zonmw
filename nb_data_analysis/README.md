Data analysis notebooks
======================
# Raw data
The notebooks that are named **raw_\*.ipynb** provide descriptive statistics about the raw data.

# Annotated data
The notebooks that are named **annot_\*.ipynb** provide descriptive statistics about the annotated data.

- For the stats of the full dataset annotated during the a-proof-zonmw phase of the project, see [annot_weeks14-34.ipynb](annot_weeks14-34.ipynb).
- For the stats of the full dataset annotated during the a-proof pilot phase, see [annot_pilot.ipynb](annot_pilot.ipynb).

# ML datasets
The notebook [datasets_for_ml.ipynb](datasets_for_ml.ipynb) provides statistics about the training, development and test sets used for the machine learning training and evaluation.

# COVID data
The notebooks that are named **covid_data_\*.ipynb** are about the COVID data of the Amsterdam UMC:

- [covid_data_stats.ipynb](covid_data_stats.ipynb) provides descriptive statistics.
- [covid_data_prep.ipynb](covid_data_prep.ipynb) adds the functioning level labels to this dataset: gold labels for the annotated notes, labels predicted by the pipeline for the non-annotated notes.
- [covid_data_annot_prep.ipynb](covid_data_annot_prep.ipynb) shows how the data for the time-series annotations (November 2021) was selected.

# UMCU data
The notebook [umcu_data_stats.ipynb](umcu_data_stats.ipynb) provides descriptive statistics about the data received from the UMCU. The labels were predicted by the pipeline.
