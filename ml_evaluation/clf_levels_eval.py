"""
Process sentence-level predictions of a regression model to generate evaluation metrics on a note-level.
The note-level labels (gold and predictions) are a mean of the sentence-level labels belonging to the same note.
The evaluation metrics include: mean absolute error, mean squared error, root mean squared error.
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

    print(f"mae: {mean_absolute_error(df.labels, df[pred_col]).round(2)}")
    print(f"mse: {mean_squared_error(df.labels, df[pred_col]).round(2)}")
    print(f"rmse: {mean_squared_error(df.labels, df[pred_col], squared=False).round(2)}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july')
    argparser.add_argument('--testdir', default='clf_levels_ADM_sents')
    argparser.add_argument('--pred_col', default='pred_levels_adm_sents')
    args = argparser.parse_args()

    test_pkl = PATHS.getpath(args.datapath) / args.testdir / 'test.pkl'

    evaluate(test_pkl, args.pred_col)