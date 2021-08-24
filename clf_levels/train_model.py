"""
Fine-tune and save regression models for levels (per domain) with Simple Transformers.

By default, models for all 9 domains are trained; if you want to only select a subset, you can pass the names of the chosen domains under the --doms parameter:

$ python train_model.py --doms ATT INS FAC
"""


import argparse
import logging
import warnings
import json
import torch
import pandas as pd
from simpletransformers.classification import ClassificationModel

import sys
sys.path.insert(0, '..')
from utils.config import PATHS


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
    argparser.add_argument('--datapath', default='data_expr_july')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--config', default='config.json')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    argparser.add_argument('--model_name', default='clin_nl_from_scratch')
    args = argparser.parse_args()

    for dom in args.doms:
        train_pkl = PATHS.getpath(args.datapath) / f"clf_levels_{dom}_sents/train.pkl"
        eval_pkl = PATHS.getpath(args.datapath) / f"clf_levels_{dom}_sents/dev.pkl"
        model_args = f"levels_{dom.lower()}_sents"
        model_name = str(PATHS.getpath(args.modelpath) / args.model_name)

        print(f"TRAINING {model_args}")
        train(
            train_pkl,
            eval_pkl,
            args.config,
            model_args,
            args.model_type,
            model_name,
        )
