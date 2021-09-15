"""
Apply a fine-tuned regression model to generate predictions.
The text is given in a pickled df and the predictions are generated per row and saved in a new 'predictions' column.

The script can be customized with the following parameters:
    --datapath: data dir
    --doms: the domains for which models are evaluated
    --model_type: type of the pre-trained model, e.g. bert, roberta, electra
    --modelpath: models dir
    --clas_unit: classification unit ('sent' or 'note')
    --pred_on: name of the file with the text

To change the default values of a parameter, pass it in the command line, e.g.:

$ python predict.py --doms ATT INS FAC --clas_unit note
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
    print("Generating predictions. This might take a while...")
    txt = df['text'].to_list()
    predictions, _ = model.predict(txt)

    col = f"pred_{Path(model_name).stem}"
    df[col] = predictions

    # pkl df
    df.to_pickle(data_pkl)
    print(f"A column with predictions was added.\nThe updated df is saved: {data_pkl}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july', help='must be listed as a key in /config.ini')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    argparser.add_argument('--clas_unit', default='sent')
    argparser.add_argument('--pred_on', default='dev')
    args = argparser.parse_args()

    for dom in args.doms:
        data_pkl = PATHS.getpath(args.datapath) / f"clf_levels_{dom}_{args.clas_unit}s/{args.pred_on}.pkl"
        model_name = PATHS.getpath(args.modelpath) / f"levels_{dom.lower()}_{args.clas_unit}s"

        predict_df(
            data_pkl,
            args.model_type,
            model_name,
        )
