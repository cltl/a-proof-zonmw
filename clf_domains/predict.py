"""
Apply a fine-tuned multi-label classification model to generate predictions.
The text is given in a pickled df and the predictions are generated per row and saved in a 'predictions' column.

The script can be customized with the following parameters:
    --datapath: data dir
    --data_pkl: the file with the text
    --model_type: type of the fine-tuned model, e.g. bert, roberta, electra
    --modelpath: models dir
    --model_name: the fine-tuned model, locally stored

To change the default values of a parameter, pass it in the command line, e.g.:

$ python predict.py --datapath data_expr_sept
"""


import argparse
import warnings
import torch
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
from pathlib import Path

import sys
sys.path.insert(0, '..')
from utils.config import PATHS
from utils.timer import timer


@timer
def predict_df(
    data_pkl,
    model_type,
    model_name,
):
    """
    Apply a fine-tuned multi-label classification model to generate predictions.
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
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
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
    argparser.add_argument('--data_pkl', default='clf_domains/test.pkl')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    argparser.add_argument('--model_name', default='domains_baseline')
    args = argparser.parse_args()

    data_pkl = PATHS.getpath(args.datapath) / args.data_pkl
    model_name = str(PATHS.getpath(args.modelpath) / args.model_name)

    predict_df(
        data_pkl,
        args.model_type,
        model_name,
    )
