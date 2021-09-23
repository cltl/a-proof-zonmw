"""
Select notes for an annotation batch, convert them to CoNLL format and save them in folders per annotator. The overview of the batch is saved as a pickled DataFrame.

The script can be customized with the following parameters:
    --datapath: path to the main directory containing all raw data
    --kwddir: path to the directory with the keyword files
    --kwdversion: the version of the keyword file to use
    --in_annot: list of paths to batch pkl's that are currently in annotation and haven't been processed yet; these notes are excluded from the selection, in addition to already annotated and processed notes
    --note_types: list of note types to select; if None, all note types are selected
    --batch: the name of the batch, as listed in the `config.batch_prep.json` file

To change the default values of a parameter, pass it in the command line, e.g.:

$ python prep_batch_for_annotation.py --note_types Consulten (niet-arts)
"""


import argparse
import json
import spacy
import pandas as pd
from pathlib import Path

from keyword_search import get_regex, get_reg_dict, find_keywords
from text_to_conll import row_to_conllfile
from select_notes import select_notes

import sys
sys.path.insert(0, '../..')
from utils.config import PATHS
from utils.df_funcs import remove_on_multikeys

HERE = Path(__file__).resolve().parent
with open(HERE / 'config.batch_prep.json', 'r', encoding='utf8') as f:
    BATCH_SETTINGS = json.load(f)


def main(
    datapath,
    kwdpath,
    in_annot,
    note_types,
    batch,
):
    """
    Select notes for an annotation batch, convert them to CoNLL format and save them in folders per annotator. The overview of the batch is saved as a pickled DataFrame.

    Parameters
    ----------
    datapath: Path
        path to raw data main folder
    kwdpath: Path
        path to the xlsx keywords file
    in_annot: list
        list of paths to batch pkl's that are currently in annotation and haven't been processed yet (these notes are excluded from the selection)
    note_types: {list, None}
        list of note types to select; if None, all note types are selected
    batch: str
        name of the batch

    Returns
    -------
    None
    """

    # load raw data
    print("Loading raw data...")
    all_2017 = pd.read_pickle(datapath / '2017_raw/processed.pkl')
    all_2018 = pd.read_pickle(datapath / '2018_raw/processed.pkl')
    all_2020 = pd.read_pickle(datapath / '2020_raw/processed.pkl')
    cov_2020 = pd.read_pickle(datapath / '2020_raw/ICD_U07.1/notes_[U07.1]_2020_q1_q2_q3.pkl')
    non_cov_2020 = remove_on_multikeys(all_2020, cov_2020, ['MDN', 'NotitieID'])
    data = {'2017': all_2017, '2018': all_2018, 'cov_2020': cov_2020, 'non_cov_2020': non_cov_2020}

    # annotated to exclude
    print("Loading annotated and 'in annotation'...")
    annotated = pd.read_csv(datapath / 'annotated_notes_ids.csv', dtype={'MDN': str, 'NotitieID': str})
    in_annotation = pd.concat([pd.read_pickle(f) for f in in_annot])
    exclude = annotated.NotitieID.append(in_annotation.NotitieID)

    # exclude annotated and sample / select specific note types
    def exclude_annotated_and_sample(df, annotated, n_sample=50000, random_state=45):
        print(f"Before exclusion: {len(df)=}")
        df = df.loc[~df.NotitieID.isin(annotated)].copy()
        print(f"After exclusion: {len(df)=}")
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=random_state)
        print(f"After sampling: {len(df)=}")
        return df

    def exclude_annotated_and_select_type(df, annotated, note_types):
        print(f"Before exclusion: {len(df)=}")
        df = df.loc[~df.NotitieID.isin(annotated)].copy()
        print(f"After exclusion: {len(df)=}")
        df = df.query(f"Typenotitie == {note_types}")
        print(f"After type selection: {len(df)=}")
        return df

    if note_types is None:
        for source, df in data.items():
            print(f"{source}:")
            data[source] = exclude_annotated_and_sample(df, exclude)
    else:
        for source, df in data.items():
            print(f"{source}:")
            data[source] = exclude_annotated_and_select_type(df, exclude, note_types=note_types)

    # keywords search
    keywords = pd.read_excel(kwdpath)
    keywords['regex'] = keywords.apply(lambda row: get_regex(row.keyword, row.regex_template_id), axis=1)
    reg_dict = get_reg_dict(keywords)

    print("Looking for keyword matches...")
    for source, df in data.items():
        data[source] = find_keywords(df, reg_dict)

    # select notes
    print("Selecting notes for the batch...")
    batch_args = BATCH_SETTINGS[batch]
    df = select_notes(data, **batch_args)

    tab = df.pivot_table(
        index=['annotator'],
        columns=['source', 'samp_meth'],
        values='NotitieID',
        aggfunc='count',
        margins=True,
        margins_name='Total',
    ).to_string()
    print(f"Batch overview:\n{tab}")

    # save batch info df
    pklpath = PATHS.getpath('data_to_inception_conll') / f"{batch}.pkl"
    df.to_pickle(pklpath)
    print(f"Batch df is saved: {pklpath}")

    # convert to conll and save in folder per annotator
    conllpath = PATHS.getpath('data_to_inception_conll')
    nlp = spacy.load('nl_core_news_sm')
    annotators = BATCH_SETTINGS[batch]["annotators"]

    for annotator in annotators:
        outdir = conllpath / batch / annotator
        outdir.mkdir(exist_ok=True, parents=True)
        print(f"Converting notes to CoNLL and saving in {outdir}")

        annot = df.query("annotator == @annotator")
        annot.apply(row_to_conllfile, axis=1, nlp=nlp, outdir=outdir, batch=batch)
        print("Done!")



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data')
    argparser.add_argument('--kwddir', default='resources_keywords')
    argparser.add_argument('--kwdversion', default='v4')
    argparser.add_argument('--in_annot', nargs='*', default=['week_22-26.pkl', 'week_27_30.pkl', 'week_31_32.pkl'])
    argparser.add_argument('--note_types', nargs='*', default=None)
    argparser.add_argument('--batch', default='week_33-34')
    args = argparser.parse_args()

    datapath = PATHS.getpath(args.datapath)
    kwdpath = PATHS.getpath(args.kwddir) / f"keywords_{args.kwdversion}.xlsx"
    in_annot_path = PATHS.getpath('data_to_inception_conll')
    in_annot = [in_annot_path / f for f in args.in_annot]

    main(datapath, kwdpath, in_annot, args.note_types, args.batch)
