"""
Select notes belonging to patients with a specific ICD_10 diagnosis."
"""


import argparse
import pandas as pd
from pathlib import Path


### ARGUMENTS ###
argparser = argparse.ArgumentParser()
argparser.add_argument('--datapath', default='../../data/2020_raw')
argparser.add_argument('--outpath', default='../../data/2020_raw/ICD_U07.1')
argparser.add_argument('--icd10', default='COVID-19, virus geïdentificeerd [U07.1]')
args = argparser.parse_args()


### PATHS ###
datapath = Path(args.datapath)
outpath = Path(args.outpath)
outpath.mkdir(exist_ok=True, parents=True)


### LOAD NOTES ###
print(f"Loading processed notes from pkl ...")
notes = pd.read_pickle(datapath / 'processed.pkl')
print(notes.institution.value_counts())


### LOAD DIAGNOSES ###
print(f"Loading all 'Diagnoses' files from {datapath}...")

cols = ['MDN', 'CSN', 'typecontact', 'DBC-id', 'specialisme', 'episodenaam', 'DBC_diagnose', 'ICD10_diagnose']
settings = dict(sep=';', names=cols, encoding = 'utf-8')
dfs = [pd.read_csv(f, **settings) for f in datapath.glob('Diagnoses*.csv')]
diag = pd.concat(dfs).astype({'MDN':str})

print(f"Diagnoses loaded: {diag.shape=}")


### SELECT AND SAVE NOTES ###
diag_code = args.icd10.split()[-1]
query = f"ICD10_diagnose == '{args.icd10}'"

crit = notes.MDN.isin(diag.query(query).MDN.unique())
selected = notes.loc[crit]

outfile = outpath / f"notes_{diag_code}_2020_q1_q2_q3.pkl"
selected.to_pickle(outfile)
print(f"Selection saved to {outfile.stem}")
