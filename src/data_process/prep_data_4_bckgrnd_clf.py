"""
The script creates a gold dataset for detecting background/target sentences. It outputs 3 pickled DataFrames: train.pkl, dev.pkl, test.pkl.

Labels:
======
1 - background/target
0 - other

Classification unit:
===================
sentence
"""


import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import ShuffleSplit


def main(annot_dir, out_dir):
    """
    Process annotations to extract 'background' and 'target' sentences. Create new sentence-level dataframes (train, dev, test) with 20% background/target sentences (label: 1) and 80% other sentences (label: 0). Save the outputs in pickle format.

    Parameters
    ----------
    annot_dir: str
        path to directory with pickled DataFrames of parsed annotations
    annot_dir: str
        path to directory to store the outputs: train.pkl, dev.pkl, test.pkl

    Returns
    -------
    None
    """
    path = Path(annot_dir)
    outpath = Path(out_dir)

    ##### LOAD DATA #####

    # load core team annotations; pickles are deduplicated during processing
    annot = pd.concat([pd.read_pickle(fp) for fp in path.glob('*_dedup.pkl')], ignore_index=True)

    # load ze annotations and remove IAA files
    ze = pd.concat(
        [pd.read_pickle(fp) for fp in path.glob('annotated_df_ze_*.pkl')], ignore_index=True
    ).query("~NotitieID.isin(@annot.NotitieID)")

    # concat and remove `disregard` files
    df = pd.concat([annot, ze], ignore_index=True).assign(
        disregard_note = lambda df: df.groupby('NotitieID').disregard.transform('max'),
    ).query("disregard_note != True").drop(columns='disregard_note')

    # load pilot annotations and remove `disregard` files
    pilot = pd.concat([pd.read_pickle(fp) for fp in path.glob('pilot_*.pkl')], ignore_index=True).assign(
        disregard_note = lambda df: df.groupby('NotitieID').disregard.transform('max'),
    ).query("disregard_note != True").drop(columns='disregard_note')

    ##### SELECT DATA #####

    zonmw_bckgrnd_sents = df.query("background | target").sen_id.unique()
    pilot_bckgrnd_sents = pilot.query("background | target").sen_id.unique()

    zonmw_bckgrnd = df.query("sen_id in @zonmw_bckgrnd_sents"
    ).groupby('sen_id'
    ).token.apply(lambda s: s.str.cat(sep=' ')).rename('text'
    ).to_frame(
    ).assign(labels=1
    ).assign(source='zonmw')

    zonmw_rest = df.query("sen_id not in @zonmw_bckgrnd_sents"
    ).groupby('sen_id'
    ).token.apply(lambda s: s.str.cat(sep=' ')).rename('text'
    ).to_frame(
    ).sample(n=20000, random_state=18
    ).assign(labels=0
    ).assign(source='zonmw')

    pilot_bckgrnd = pilot.query("sen_id in @pilot_bckgrnd_sents"
    ).groupby('sen_id'
    ).token.apply(lambda s: s.str.cat(sep=' ')).rename('text'
    ).to_frame(
    ).assign(labels=1
    ).assign(source='pilot')

    pilot_rest = pilot.query("sen_id not in @pilot_bckgrnd_sents"
    ).groupby('sen_id'
    ).token.apply(lambda s: s.str.cat(sep=' ')).rename('text'
    ).to_frame(
    ).sample(n=180000, random_state=18
    ).assign(labels=0
    ).assign(source='pilot')

    data = zonmw_bckgrnd.append([zonmw_rest, pilot_bckgrnd, pilot_rest])

    print(f"""
    Selected data:\n\n
    {data.pivot_table(
        index=['source', 'labels'],
        values='text',
        aggfunc='count',
        margins=True,
        margins_name='total',
    ).rename(columns={'text': 'n_sentences'})}
    """)

    ##### SPLIT DATA #####

    # 70% of the data goes into train df
    ss = ShuffleSplit(n_splits=1, test_size=0.3, random_state=19)
    for train_idx, other_idx in ss.split(data.text, data.labels):
        train = data.iloc[train_idx]
        other = data.iloc[other_idx]

    # the non-train data is split 50/50 into development df and test df
    ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for dev_idx, test_idx in ss.split(other.text, other.labels):
        dev = other.iloc[dev_idx]
        test = other.iloc[test_idx]

    def make_table(df):
        return df.pivot_table(
            index=['source', 'labels'],
            values='text',
            aggfunc='count',
            margins=True,
            margins_name='total',
        )

    tab_train = make_table(train).rename(columns={'text': 'train'})
    tab_dev = make_table(dev).rename(columns={'text': 'dev'})
    tab_test = make_table(test).rename(columns={'text': 'test'})

    tab = tab_train.join([tab_dev, tab_test])
    print(f"Split data:\n\n{tab}")

    ##### SAVE #####

    train.to_pickle(outpath / 'train.pkl')
    dev.to_pickle(outpath / 'dev.pkl')
    test.to_pickle(outpath / 'test.pkl')

    print(f"Data saved in {outpath}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--annot_dir', default='../../data/expr_july')
    argparser.add_argument('--out_dir', default='../../data/expr_july/bckgrnd_clf')
    args = argparser.parse_args()

    main(
        args.annot_dir,
        args.out_dir,
    )
