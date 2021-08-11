"""
Prepare and save train, test, dev data for a sentence-level regression model.

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
from utils.data_process import concat_annotated, drop_disregard, fix_week_14, pad_sen_id, anonymize, data_split_groups


def prep_levels_for_dom(
    df,
    dom,
    outdir,
    domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Prepare and save datasets (all, train, test, dev) of levels for a specific domain (`dom`). The pickled datasets, as well as a figure with the labels distribution, are saved in `outdir`.
    The data for the chosen domain consists of all sentences containing level labels of this domain (e.g. ETN_lvl); the sentence-level label is the mean of all the level labels (of this domain) in the sentence.

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

    print(f"###### PROCESSING {dom}_lvl ######")

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

    # save fig with the distribution of labels
    figpath = outdir / f"labels_hist.png"
    fig, ax = plt.subplots()
    dom_df.labels.plot(kind='hist', grid=True, ax=ax)
    fig.savefig(figpath)
    print(f"Labels distribtion fugure saved to: {figpath}")

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

    # train / dev / test split
    print("Splitting data to train / dev / test...")
    train, dev, test = data_split_groups(
        dom_df,
        'text',
        'labels',
        'NotitieID',
        0.8,
    )
    print("Done!")
    print(f"{len(train) = }")
    print(f"{len(dev) = }")
    print(f"{len(test) = }")
    
    # save
    dom_df.to_pickle(outdir / 'all.pkl')
    train.to_pickle(outdir / 'train.pkl')
    dev.to_pickle(outdir / 'dev.pkl')
    test.to_pickle(outdir / 'test.pkl')
    print(f"The pickles are saved at: {outdir}")


def main(
    datapath,
    doms,
    domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """

    Parameters
    ----------
    datapath: Path
        path to directory with parsed annotations in pkl format
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

    # data prep per domain
    for dom in doms:
        outdir = datapath / f"clf_levels_{dom}_sents"
        prep_levels_for_dom(df, dom, outdir)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    args = argparser.parse_args()

    datapath = PATHS.getpath(args.datapath)
    
    main(datapath, args.doms)