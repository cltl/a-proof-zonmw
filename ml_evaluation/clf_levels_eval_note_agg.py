"""
Process sentence-level predictions of a regression model to generate evaluation metrics on a note-level.
The note-level labels (gold and predictions) are a mean of the sentence-level labels belonging to the same note.
The evaluation metrics include: mean absolute error, mean squared error, root mean squared error.

By default, metrics are generated for all 9 domains; if you want to only select a subset, you can pass the names of the chosen domains under the --doms parameter:

$ python evaluate_model.py --doms ATT INS FAC
"""


import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, '..')
from utils.config import PATHS


def evaluate(test_pkl, pred_col):
    """
    Process sentence-level predictions of a regression model to generate evaluation metrics on a note-level.
    The note-level labels (gold and predictions) are a mean of the sentence-level labels belonging to the same note.
    The evaluation metrics include: mean absolute error, mean squared error, root mean squared error. The values are printed to the command line.

    Parameters
    ----------
    test_pkl: str
        path to pickled df with the training data, which must contain a 'labels' column and a column with predictions (whose name is given by the `pred_col` argument); both columns contain numeric values on a continuous scale
    pred_col: str
        the name of the column containing the predictions; its format is "preds_{name_of_the_model}"

    Returns
    -------
    None
    """

    # load data
    test = pd.read_pickle(test_pkl)

    # aggregate on note level
    labels = test.groupby('NotitieID').labels.mean()
    preds = test.groupby('NotitieID')[pred_col].mean()
    df = pd.concat([labels, preds], axis=1)
    print(f"Number of notes in the test set: {len(df)}")
    df = df.dropna()
    print(f"Number of notes with a gold label: {len(df)}")
    
    print(f"mae: {mean_absolute_error(df.labels, df[pred_col]).round(2)}")
    print(f"mse: {mean_squared_error(df.labels, df[pred_col]).round(2)}")
    print(f"rmse: {mean_squared_error(df.labels, df[pred_col], squared=False).round(2)}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--testfile', default='test')
    args = argparser.parse_args()

    for dom in args.doms:
        test_pkl = PATHS.getpath(args.datapath) / f"clf_levels_{dom}_sents/{args.testfile}.pkl"
        pred_col = f"pred_levels_{dom.lower()}_sents"

        print(f"Note-level metrics for {dom}_{args.testfile}:")
        evaluate(test_pkl, pred_col)