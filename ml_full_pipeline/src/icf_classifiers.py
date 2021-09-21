"""
Functions for generating predictions:

- `predict_domains` generates a multi-label which indicates which of the 9 ICF domains are discussed in a given sentence; the order is ['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'], i.e. if the sentence is labeled as [1, 0, 0, 0, 0, 1, 0, 0, 0], it means it contains the ADM and FAC domains

- `predict_levels` generates a float that indicates the level of functioning (for a specific domain) discussed in the sentence
"""


import pandas as pd
import torch
import warnings
from simpletransformers.classification import MultiLabelClassificationModel, ClassificationModel


def predict_domains(
    text,
    model_type,
    model_name,
):
    """
    Apply a fine-tuned multi-label classification model to generate predictions.

    Parameters
    ----------
    text: pd Series
        a series of strings
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: {str, Path}
        path to a directory containing model file

    Returns
    -------
    df: pd Series
        a series of lists; each list is a multi-label prediction
    """

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
    predictions, _ = model.predict(text.to_list())
    return pd.Series(predictions, index=text.index)


def predict_levels(
    text,
    model_type,
    model_name,
):
    """
    Apply a fine-tuned regression model to generate predictions.

    Parameters
    ----------
    text: pd Series
        a series of strings
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: {str, Path}
        path to a directory containing model file

    Returns
    -------
    predictions: pd Series
        a series of floats or an empty series (if text is empty)
    """

    to_predict = text.to_list()
    if not len(to_predict):
        return pd.Series()

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
    predictions, _ = model.predict(to_predict)
    return pd.Series(predictions, index=text.index)
