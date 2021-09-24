Process raw data
===============
# From csv to DataFrame
The [preprocess_raw_data.py](preprocess_raw_data.py) script processes raw data csv's in a given directory and saves the combined data in a pickled DataFrame.

Processing steps:
1. combine all csv files from the same institution
2. drop any duplicated rows
3. concatinate all text columns into one string (`all_text`)
4. clean up text: remove whitespace and strip string
5. convert MDN and NotitieID to str
6. add institution
7. combine df's per institution into one df

The default settings for the `pd.read_csv` function are:
- sep = ;
- encoding = utf-8-sig
- no header

These defaults can be overwritten by the settings in the [config.data_load.json](config.data_load.json); this is necessary due to differences in raw data formats.

# Select data by ICD_10 diagnosis
The script [select_data_by_diag.py](select_data_by_diag.py) selects notes belonging to patients with a specific ICD_10 diagnosis and saves them to a pickled dataframe.
