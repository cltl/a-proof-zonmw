"""
Functions used in pre-processing of data for the machine learning pipelines.
"""


import pandas as pd
from pandas.api.types import is_scalar
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def concat_annotated(datadir):
    """
    Concatenate all "annotated_df_*_parsed*.pkl" files in `datadir`.
    The pkl's of the core team should end with "dedup.pkl", i.e. they should be deduplicated by the `parse_annotations.py` script.
    The ze pkl's need not be deduplicated, as only notes that are not in the annotations of the core team are included.

    Parameters
    ----------
    datadir: Path
        path to directory with data

    Returns
    -------
    DataFrame
        df of concatenated parsed annotations
    """

    # load core team annotations; pickles are deduplicated during processing
    annot = pd.concat([pd.read_pickle(fp) for fp in datadir.glob('*_dedup.pkl')], ignore_index=True)

    # load ze annotations and remove IAA files
    ze = pd.concat(
        [pd.read_pickle(fp) for fp in datadir.glob('annotated_df_ze_*.pkl')], ignore_index=True
    ).query("~NotitieID.isin(@annot.NotitieID)", engine='python')

    return pd.concat([annot, ze], ignore_index=True)


def drop_disregard(df):
    """
    If one token in a note is marked 'disregard', remove the whole note from df.

    Parameters
    ----------
    df: DataFrame
        parsed token-level annotations df (created by `parse_annotations.py`)

    Returns
    -------
    DataFrame
        df without 'disregard' notes
    """

    df['disregard_note'] = df.groupby('NotitieID').disregard.transform('any')

    return df.query(
        "not disregard_note"
    ).drop(columns=['disregard', 'disregard_note'])


def fix_week_14(df):
    """
    For annotations from week 14:
    - Replace MBW values with `False`
    - Replace MBW-lvl values with NaN
    We remove this domain from week 14 since the guidelines for it were changed after this week.

    Parameters
    ----------
    df: DataFrame
        parsed token-level annotations df (created by `parse_annotations.py`)

    Returns
    -------
    DataFrame
        df without MBW and MBW_lvl labels for week 14
    """

    df['MBW'] = df.MBW.mask(df.batch == 'week_14', other=False)
    df['MBW_lvl'] = df.MBW_lvl.mask(df.batch == 'week_14')

    return df


def pad_sen_id(id):
    """
    Add padding zeroes to sen_id.
    """
    note_id, sen_no = id.split('_')
    return '_'.join([note_id, f"{sen_no:0>4}"])


def anonymize(txt, nlp):
    """
    Replace entities of type PERSON and GPE with 'PERSON', 'GPE'.
    Return anonymized text and its length.
    """
    doc = nlp(txt)
    anonym = str(doc)
    to_repl = {str(ent):ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'GPE']}
    for string, replacement in to_repl.items():
        anonym = anonym.replace(string, replacement)
    return anonym, len(doc)


def data_split_groups(
    df,
    X_col,
    y_col,
    group_col,
    train_size,
):
    """
    Split data to train / dev / test, while taking into account groups that should stay together.

    Parameters
    ----------
    df: DataFrame
        df with the data to split
    X_col: str
        name of the column with the data (text)
    y_col: str
        name of the column with the gold labels
    group_col: str
        name of the column with the groups to take into account when splitting
    train_size: float
        proportion of data that should go to the training set
    
    Returns
    -------
    train, dev, test: DataFrame's
        df with train data, df with dev data, df with test data
    """
    # create training set of `train_size`
    gss = GroupShuffleSplit(n_splits=1, test_size=1-train_size, random_state=19)
    for train_idx, other_idx in gss.split(df[X_col], df[y_col], groups=df[group_col]):
        train = df.iloc[train_idx]
        other = df.iloc[other_idx]

    # the non-train data is split 50/50 into development and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=19)
    for dev_idx, test_idx in gss.split(other[X_col], other[y_col], groups=other[group_col]):
        dev = other.iloc[dev_idx]
        test = other.iloc[test_idx]
    
    return train, dev, test


def flatten_preds_if_necessary(df):
    """
    Flatten predictions if they are a list in a list.
    This is necessary because of an issue with the predict.py script prior to the update performed on 15-09-2021.
    """
    cols = [col for col in df.columns if 'pred' in col]
    for col in cols:
        test = df[col].iloc[0]
        if is_scalar(test[0]):
            continue
        df[col] = df[col].str[0]
    return df