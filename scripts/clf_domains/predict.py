"""
Apply a fine-tuned multi-label classification model to generate predictions.
The text is given in a pickled df and the predictions are generated per row and saved in a 'predictions' column.
"""


import argparse
import warnings
import torch
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel


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
    def predict(txt):
        predictions, _ = model.predict([txt])
        return predictions

    df['predictions'] = df['text'].apply(predict)

    # pkl df
    df.to_pickle(data_pkl)
    print(f"A column with predictions was added.\nThe updated df is saved: {data_pkl}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_pkl', default='../../data/expr_july/clf_domains/data.pkl')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--model_name', default='../../models/domains_spacy_default')
    args = argparser.parse_args()

    predict_df(
        args.data_pkl,
        args.model_type,
        args.model_name,
    )
