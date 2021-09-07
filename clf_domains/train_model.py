"""
Fine-tune and save a multi-label classification model with Simple Transformers.
"""


import argparse
import logging
import warnings
import json
import torch
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel

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
    labels=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Fine-tune and save a multi-label classification model with Simple Transformers.

    Parameters
    ----------
    train_pkl: str
        path to pickled df with the training data, which must contain the columns 'text' and 'labels'; the labels are multi-hot lists (see column indices in `labels`), e.g. [1, 0, 0, 1, 0, 0, 0, 0, 1]
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
    labels: list
        list of column indices for the multi-hot labels

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
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
        num_labels=len(labels),
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
    argparser.add_argument('--train_pkl', default='clf_domains/train.pkl')
    argparser.add_argument('--eval_pkl', default='clf_domains/dev.pkl')
    argparser.add_argument('--model_config', default='config.json')
    argparser.add_argument('--model_args', default='domains_baseline')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    argparser.add_argument('--model_name', default='clin_nl_from_scratch')
    argparser.add_argument('--hf', dest='hugging_face', action='store_true')
    argparser.set_defaults(hugging_face=False)
    args = argparser.parse_args()

    train_pkl = PATHS.getpath(args.datapath) / args.train_pkl
    eval_pkl = PATHS.getpath(args.datapath) / args.eval_pkl
    
    # model stored locally (default) or on HuggingFace (--hf)
    model_name = str(PATHS.getpath(args.modelpath) / args.model_name)
    if args.hugging_face:
        model_name = args.model_name
    

    train(
        train_pkl,
        eval_pkl,
        args.model_config,
        args.model_args,
        args.model_type,
        model_name,
    )
