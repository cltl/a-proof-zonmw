"""
Select notes belonging to patients with a specific ICD_10 diagnosis."
"""


import argparse
import pandas as pd
from pathlib import Path


### ARGUMENTS ###
argparser = argparse.ArgumentParser()
argparser.add_argument('--datapath', default='../../Covid_data_11nov/raw/')
argparser.add_argument('--outpath', default='../../Covid_data_11nov/raw/ICD_U07.1/a-proof-zonmw/')
argparser.add_argument('--icd10', default='COVID-19, virus geïdentificeerd [U07.1]')
args = argparser.parse_args()


### PATHS ###
datapath = Path(args.datapath)
outpath = Path(args.outpath)
outpath.mkdir(exist_ok=True, parents=True)


### LOAD NOTES ###
print(f"Loading all 'Notities' files from {datapath}...")

cols = ['MDN', 'NotitieID', 'NotitieCSN', 'Typenotitie', 'Notitiedatum', 'Notitietekst1', 'Notitietekst2', 'Notitietekst3']
amc = pd.concat(pd.read_csv(f, sep=';', names=cols, encoding='utf-8-sig') for f in datapath.glob('Notities AMC*.csv'))
vumc = pd.concat(pd.read_csv(f, sep=';', names=cols, encoding='utf-8-sig') for f in datapath.glob('Notities VUMC*.csv'))

print(f"DataFrames loaded: {amc.shape=}, {vumc.shape=}")


### LOAD DIAGNOSES ###
print(f"Loading all 'Diagnoses' files from {datapath}...")

cols = ['MDN', 'CSN', 'typecontact', 'DBC-id', 'specialisme', 'episodenaam', 'DBC_diagnose', 'ICD10_diagnose']
f = datapath / 'Diagnoses AMC 2020 sept.csv'
diag_amc = pd.read_csv(f, sep=';', names=cols, encoding = 'utf-8')
f = datapath / 'Diagnoses VUMC 2020 sept.csv'
diag_vumc = pd.read_csv(f, sep=';', names=cols, encoding = 'utf-8')

print(f"DataFrames loaded: {diag_amc.shape=}, {diag_vumc.shape=}")


### SELECT AND SAVE NOTES ###
diag_code = args.icd10.split()[-1]
query = f"ICD10_diagnose == '{args.icd10}'"

def select_notes(df, diag_df, query):
    crit = df.MDN.isin(diag_df.query(query).MDN.unique())
    subset = ['NotitieID', 'Notitietekst1', 'Notitietekst2', 'Notitietekst3']
    return df.loc[crit].drop_duplicates(subset=subset, keep='first')

def save_results(hospital, df, diag_code):
    outfile = outpath / f"{hospital}_{diag_code}_2020_q1_q2_q3.csv"
    df.to_csv(outfile, index_label='idx_source_file')
    print(f"Number patients with {diag_code} diagnosis in {hospital}: {df.MDN.nunique()}")
    print(f"Number notes belonging to these patients: {df.NotitieID.nunique()}")
    print(f"Selection saved to {outfile}")
    return None

amc_notes_diag = select_notes(amc, diag_amc, query)
save_results('amc', amc_notes_diag, diag_code)

vumc_notes_diag = select_notes(vumc, diag_vumc, query)
save_results('vumc', vumc_notes_diag, diag_code)
