"""
Perform a sweep for hyperparameters optimization, using Simple Transformers and W&B Sweeps.
The sweep is configured in a dictionary in a config file, which should specify the search strategy, the metric to be optimized, and the hyperparameters (and their possible values).

The script can be customized with the following parameters:
    --datapath: data dir
    --train_pkl: the file with the train data
    --eval_pkl: the file with the eval data
    --config: json file containing the model args
    --sweep_config: the name of the sweep config dict from `config`
    --model_args: the name of the model args dict from `config`
    --model_type: type of the pre-trained model, e.g. bert, roberta, electra
    --modelpath: models dir
    --model_name: the pre-trained model, either from Hugging Face or locally stored
    --hf: pass this parameter if a model from Hugging Face is used

To change the default values of a parameter, pass it in the command line, e.g.:

$ python wandb_sweep.py --model_name pdelobelle/robbert-v2-dutch-base --hf
"""


import argparse
import warnings
import json
import torch
import wandb
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel

import sys
sys.path.insert(0, '..')
from utils.config import PATHS


def main(
    train_pkl,
    eval_pkl,
    config_json,
    sweep_config,
    model_args,
    model_type,
    model_name,
    labels=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Perform a sweep for hyperparameters optimization, using Simple Transformers and W&B Sweeps.
    The sweep is configured in a dictionary in `config_json`, which should specify the search strategy, the metric to be optimized, and the hyperparameters (and their possible values).

    Parameters
    ----------
    train_pkl: str
        path to pickled df with the training data, which must contain the columns 'text' and 'labels'; the labels are multi-hot lists (see column indices in `labels`), e.g. [1, 0, 0, 1, 0, 0, 0, 0, 1]
    eval_pkl: str
        path to pickled df for evaluation during training
    config_json: str
        path to a json file containing the sweep config
    sweep_config: str
        the name of the sweep config dict from `config_json` to use
    model_args: str
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

    # load data
    train_data = pd.read_pickle(train_pkl)
    eval_data = pd.read_pickle(eval_pkl)

    # sweep config & model args
    with open(config_json, 'r') as f:
        config = json.load(f)
    sweep_config = config[sweep_config]
    model_args = config[model_args]

    sweep_id = wandb.sweep(sweep_config, project=model_args['wandb_project'])

    def train():
        wandb.init()

        model = MultiLabelClassificationModel(
            model_type,
            model_name,
            num_labels=len(labels),
            args=model_args,
            use_cuda=cuda_available,
        )

        model.train_model(train_data, eval_df=eval_data)

        wandb.join()

    wandb.agent(sweep_id, train)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july', help='must be listed as a key in /config.ini')
    argparser.add_argument('--train_pkl', default='clf_domains/train.pkl')
    argparser.add_argument('--eval_pkl', default='clf_domains/dev.pkl')
    argparser.add_argument('--config', default='config.json')
    argparser.add_argument('--sweep_config', default='sweep_config')
    argparser.add_argument('--model_args', default='sweep_args')
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

    main(
        train_pkl,
        eval_pkl,
        args.config,
        args.sweep_config,
        args.model_args,
        args.model_type,
        model_name,
    )
