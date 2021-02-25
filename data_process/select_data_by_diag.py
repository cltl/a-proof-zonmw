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

cols = ['MDN',  'NotitieID',  'NotitieCSN', 'Typenotitie',  'Notitiedatum', 'Notitietekst1', 'Notitietekst2', 'Notitietekst3']
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

# AMC
# select
amc_notes_diag = amc.loc[amc.MDN.isin(diag_amc.query(query).MDN.unique())]
# save
outfile = outpath / f"amc_{diag_code}_2020_q1_q2_q3.csv"
amc_notes_diag.to_csv(outfile, index_label='idx_source_file')

print(f"Number patients with {diag_code} diagnosis in AMC: {diag_amc.query(query).MDN.nunique()}")
print(f"Number notes belonging to these patients: {amc_notes_diag.shape[0]}")
print(f"Selection saved to {outfile}")

# VUMC
#select
vumc_notes_diag = vumc.loc[vumc.MDN.isin(diag_vumc.query(query).MDN.unique())]
# save
outfile = outpath / f"vumc_{diag_code}_2020_q1_q2_q3.csv"
vumc_notes_diag.to_csv(outfile, index_label='idx_source_file')

print(f"Number patients with {diag_code} diagnosis in VUMC: {diag_vumc.query(query).MDN.nunique()}")
print(f"Number notes belonging to these patients: {vumc_notes_diag.shape[0]}")
print(f"Selection saved to {outfile}")
