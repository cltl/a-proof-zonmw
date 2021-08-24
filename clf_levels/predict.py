"""
Apply a fine-tuned regression model to generate predictions.
The text is given in a pickled df and the predictions are generated per row and saved in a new 'predictions' column.

By default, predictions are generated for all 9 domains; if you want to only select a subset, you can pass the names of the chosen domains under the --doms parameter:

$ python evaluate_model.py --doms ATT INS FAC
"""


import argparse
import warnings
import torch
import pandas as pd
from simpletransformers.classification import ClassificationModel
from pathlib import Path

import sys
sys.path.insert(0, '..')
from utils.config import PATHS


def predict_df(
    data_pkl,
    model_type,
    model_name,
):
    """
    Apply a fine-tuned regression model to generate predictions.
    The text is given in `data_pkl` and the predictions are generated per row and saved in a 'predictions' column.

    Parameters
    ----------
    data_pkl: str
        path to pickled df with the data, which must contain the column 'text'
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        path to a directory containing model file

    Returns
    -------
    None
    """

    # load data
    df = pd.read_pickle(data_pkl)

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn('CUDA device not available; running on a CPU!')

    # load model
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=1,
        use_cuda=cuda_available,
    )

    # predict
    def predict(txt):
        predictions, _ = model.predict([txt])
        return predictions

    col = f"pred_{Path(model_name).stem}"
    df[col] = df['text'].apply(predict)

    # pkl df
    df.to_pickle(data_pkl)
    print(f"A column with predictions was added.\nThe updated df is saved: {data_pkl}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    args = argparser.parse_args()

    for dom in args.doms:
        data_pkl = PATHS.getpath(args.datapath) / f"clf_levels_{dom}_sents/test_dom_output.pkl"
        model_name = PATHS.getpath(args.modelpath) / f"levels_{dom.lower()}_sents"

    data_pkl = PATHS.getpath(args.datapath) / args.data_pkl
    model_name = str(PATHS.getpath(args.modelpath) / args.model_name)

    predict_df(
        data_pkl,
        args.model_type,
        model_name,
    )
