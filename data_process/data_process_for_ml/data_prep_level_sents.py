"""
Prepare and save train, test, dev data for a sentence-level regression model for levels (per domain).

The data split is based on the split used for the domains classifier, i.e. the notes that were in the test set of the domains classifier are in the test set for the levels classifiers as well.

By default, data is prepped for all 9 domains; if you want to only select a subset, you can pass the names of the chosen domains under the --doms parameter:

$ python data_prep_level_sents.py --datapath my/data/ --doms ATT INS FAC
"""


import argparse
import spacy
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../..')
from utils.config import PATHS
from utils.data_process import concat_annotated, drop_disregard, fix_week_14, pad_sen_id, anonymize


def prep_gold_data_for_dom(
    df,
    dom,
    outdir,
    domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Select from `df` the sentences containing a gold level label for the chosen domain (`dom`); the sentence-level label is the mean of all the level labels (of this domain) in the sentence.
    Anonynimyze the text and save the final pickled dataset in `outdir`.

    Parameters
    ----------
    df: DataFrame
        sentence-level df with the sentence-level labels in "{dom}_lvl" column
    dom: str
        the domain for which the data is prepped
    outdir: Path
        path to directory where the outputs are saved
    domains: list
        a list of all the domains

    Returns
    -------
    None
    """

    print(f"###### PROCESSING {df.name} for {dom}_lvl ######")

    # check path
    try:
        outdir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{outdir} exists, all good.")
    else:
        print(f"{outdir} was created.")
    
    # select sentences for the chosen domain
    cols_to_drop = [f"{domain}_lvl" for domain in domains if domain != dom]
    lvl = f"{dom}_lvl"

    dom_df = df.loc[df[lvl].notna()].drop(columns=cols_to_drop
    ).rename(columns={lvl: 'labels'})

    print(f"Number of sentences with {lvl}: {len(dom_df)}")

    # anonymize text
    print("Anonymizing text...")
    nlp = spacy.load('nl_core_news_lg')
    dom_df = dom_df.join(
        dom_df.apply(
            lambda row: anonymize(row.text_raw, nlp),
            axis=1,
            result_type='expand',
        ).rename(columns={0: 'text', 1: 'len_text'})
    )
    print("Done!")
    print(f"Text length per sentence: {dom_df.len_text.agg(['min', 'max', 'median', 'mean']).astype(int)}")
    
    # save
    dom_df.to_pickle(outdir / f'{df.name}.pkl')
    print(f"The pickle is saved at: {outdir}")


def prep_test_dom_output(
    df,
    predictions,
    dom,
    outdir,
    domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Select from `df` the sentences that the domains classifier labeled as `dom`.
    If the prediction coincides with the gold annotation, the sentence-level label is the mean of all the level labels (of this domain) in the sentence. If the prediction is a false positive, then the sentence does not have a level label (NaN).
    Anonynimyze the text and save the final pickled dataset in `outdir`.

    Parameters
    ----------
    df: DataFrame
        sentence-level df with the sentence-level labels in "{dom}_lvl" column
    predictions: DataFrame
        df with the columns 'pad_sen_id', 'domains', 'preds'; each row contains a prediction per domain per sentence (1 or 0)
    dom: str
        the domain for which the data is prepped
    outdir: Path
        path to directory where the outputs are saved
    domains: list
        a list of all the domains

    Returns
    -------
    None
    """

    print(f"###### PROCESSING `test_dom_output` for {dom}_lvl ######")

    # check path
    try:
        outdir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{outdir} exists, all good.")
    else:
        print(f"{outdir} was created.")
    
    # select sentences for the chosen domain
    cols_to_drop = [f"{domain}_lvl" for domain in domains if domain != dom]
    lvl = f"{dom}_lvl"
    preds = predictions.query("domains == @dom and preds == 1").pad_sen_id

    dom_df = df.query("pad_sen_id in @preds"
    ).drop(columns=cols_to_drop
    ).rename(columns={lvl: 'labels'})

    if dom_df.empty:
        print(f"No {dom} predictions found! Moving to the next domain...")
        return None
    
    print(f"Number of sentences with {lvl}: {len(dom_df)}")

    # anonymize text
    print("Anonymizing text...")
    nlp = spacy.load('nl_core_news_lg')
    dom_df = dom_df.join(
        dom_df.apply(
            lambda row: anonymize(row.text_raw, nlp),
            axis=1,
            result_type='expand',
        ).rename(columns={0: 'text', 1: 'len_text'})
    )
    print("Done!")
    print(f"Text length per sentence: {dom_df.len_text.agg(['min', 'max', 'median', 'mean']).astype(int)}")
    
    # save
    dom_df.to_pickle(outdir / 'test_dom_output.pkl')
    print(f"The pickle is saved at: {outdir}")


def main(
    datapath,
    doms_train,
    doms_test,
    doms_dev,
    pred_col,
    doms,
    domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Prepare and save train, test, dev data for a sentence-level regression model per domain in `doms`.

    Two test sets are created for each domain: one that contains the sentences that have gold levels labels (test.pkl), and one that contains the sentences that were assigned a domain label by the domains classifier and do not necessarily have a gold label (test_dom_output.pkl).

    The data split is based on the split used for the domains classifier, i.e. the notes that were in the test set of the domains classifier are in the test set for the levels models as well.

    Parameters
    ----------
    datapath: Path
        path to directory with parsed annotations in pkl format
    doms_train: Path
        path to the train data (pkl) for the domains classifier
    doms_test: Path
        path to the test data (pkl) for the domains classifier
    doms_dev: Path
        path to the dev data (pkl) for the domains classifier
    pred_col: str
        name of the column in `doms_test` from which predictions are taken
    doms: list
        the domains for which the data is prepped
    domains: list
        a list of all the domains

    Returns
    -------
    None
    """

    # labels column names
    levels = [f"{domain}_lvl" for domain in domains]
    other = ['target', 'background', 'plus']

    # load and pre-process data
    print(f"Pre-processing data in {datapath}...")
    df = concat_annotated(datapath
        ).pipe(drop_disregard
        ).pipe(fix_week_14)
    
    # sentence-level pre-process
    df = df.assign(
        background_sent = lambda df: df.groupby('sen_id').background.transform('any'),
        target_sent = lambda df: df.groupby('sen_id').target.transform('any'),
        pad_sen_id = df.sen_id.apply(pad_sen_id)
    )

    # fill NA
    df[domains + other] = df[domains + other].fillna(False)
    df[['label', 'relation']] = df[['label', 'relation']].fillna('_')
    df['token'] = df['token'].fillna('')

    # create sentence-level df
    print("Creating sentence-level df...")
    info_cols = ['pad_sen_id', 'institution', 'year', 'MDN', 'NotitieID', 'batch', 'annotator', 'background_sent', 'target_sent']

    info = df.groupby('pad_sen_id')[info_cols].first()
    text = df.groupby('pad_sen_id').token.apply(lambda s: s.str.cat(sep=' ')).rename('text_raw')
    labels = df.groupby('pad_sen_id')[levels].mean()
    df = pd.concat([info, text, labels], axis=1)

    # data split
    print("Splitting train / test / dev...")
    doms_train = pd.read_pickle(doms_train)
    doms_test = pd.read_pickle(doms_test)
    doms_dev = pd.read_pickle(doms_dev)

    test = df.query("NotitieID in @doms_test.NotitieID.unique()")
    dev = df.query("NotitieID in @doms_dev.NotitieID.unique()")
    train = df.query("NotitieID not in @test.NotitieID and NotitieID not in @dev.NotitieID")

    test.name = 'test'
    dev.name = 'dev'
    train.name = 'train'

    # data prep per domain
    # train, dev, test
    for dom in doms:
        for df in [train, dev, test]:
            outdir = datapath / f"clf_levels_{dom}_sents"
            prep_gold_data_for_dom(df, dom, outdir)
    
    # test_doms_output
    doms_test = doms_test.assign(
        preds = lambda df: df[pred_col].str[0],
        domains = lambda df: [domains] * len(df),
    )
    predictions = doms_test.explode(
        ['domains', 'labels', 'preds']
    )[['pad_sen_id', 'NotitieID', 'annotator', 'domains', 'labels', 'preds']].reset_index()

    for dom in doms:
        outdir = datapath / f"clf_levels_{dom}_sents"
        prep_test_dom_output(test, predictions, dom, outdir)




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july')
    argparser.add_argument('--doms_train', default='clf_domains/train_excl_bck.pkl')
    argparser.add_argument('--doms_test', default='clf_domains/test.pkl')
    argparser.add_argument('--doms_dev', default='clf_domains/dev.pkl')
    argparser.add_argument('--pred_col', default='pred_domains_excl_bck')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    args = argparser.parse_args()

    datapath = PATHS.getpath(args.datapath)
    doms_train = datapath / args.doms_train
    doms_test = datapath / args.doms_test
    doms_dev = datapath / args.doms_dev
    
    main(
        datapath,
        doms_train,
        doms_test,
        doms_dev,
        args.pred_col,
        args.doms,
    )