"""
Fine-tune and save a regression model with Simple Transformers.
"""


import argparse
import logging
import warnings
import json
import torch
import pandas as pd
from simpletransformers.classification import ClassificationModel


def train(
    train_pkl,
    eval_pkl,
    config_json,
    args,
    model_type,
    model_name,
):
    """
    Fine-tune and save a regression model with Simple Transformers.

    Parameters
    ----------
    train_pkl: str
        path to pickled df with the training data, which must contain the columns 'text' and 'labels'; the labels are numeric values on a continuous scale
    eval_pkl: {None, str}
        path to pickled df for evaluation during training (optional)
    config_json: str
        path to a json file containing the model args
    args: str
        the name of the model args dict from `config_json` to use
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        the exact architecture and trained weights to use; this can be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model file

    Returns
    -------
    None
    """

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn('CUDA device not available; running on a CPU!')

    # logging
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.WARNING)

    # load data
    train_data = pd.read_pickle(train_pkl)
    eval_data = pd.read_pickle(eval_pkl)

    # model args
    with open(config_json, 'r') as f:
        config = json.load(f)
    model_args = config[args]

    # model
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=1,
        args=model_args,
        use_cuda=cuda_available,
    )

    # train
    if model.args.evaluate_during_training:
        model.train_model(train_data, eval_df=eval_data)
    else:
        model.train_model(train_data)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_pkl', default='../../data/expr_july/clf_levels_ADM/train.pkl')
    argparser.add_argument('--eval_pkl', default='../../data/expr_july/clf_levels_ADM/dev.pkl')
    argparser.add_argument('--config', default='config.json')
    argparser.add_argument('--model_args', default='levels_adm_notes')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--model_name', default='../../models/clin_nl_from_scratch/')
    args = argparser.parse_args()

    train(
        args.train_pkl,
        args.eval_pkl,
        args.config,
        args.model_args,
        args.model_type,
        args.model_name,
    )
