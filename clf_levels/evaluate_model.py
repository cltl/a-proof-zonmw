"""
Evaluate fine-tuned regression models on two test sets:
- sentences with gold levels labels (test.pkl)
- sentences that were assigned a domain label by the domains classifier and do not necessarily have a gold label (test_dom_output.pkl)

Save the following outputs per model:
- evaluation metrics: MSE, RMSE, MAE, eval_loss
- model outputs
- wrong predictions

By default, models for all 9 domains are evaluated; if you want to only select a subset, you can pass the names of the chosen domains under the --doms parameter:

$ python evaluate_model.py --doms ATT INS FAC
"""


import argparse
import pickle
import warnings
import torch
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, '..')
from utils.config import PATHS


def evaluate(
    test_pkl,
    model_type,
    model_name,
    output_dir,
):
    """
    Evaluate a fine-tuned regression model on a test set.
    Save evaluation metrics, model outputs and wrong predictions in `output_dir`. The evaluation metrics include: MSE, RMSE, MAE and eval_loss.

    Parameters
    ----------
    test_pkl: str
        path to pickled df with the test data, which must contain the columns 'text' and 'labels'; the labels are numeric values on a continuous scale
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        path to a directory containing model file
    output_dir: Path
        path to a directory where outputs should be saved

    Returns
    -------
    None
    """

    # load data
    test_data = pd.read_pickle(test_pkl)

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

    # evaluate model
    result, model_outputs, wrong_predictions = model.eval_model(
        test_data,
        output_dir=str(output_dir),
        mae=mean_absolute_error,
        mse=mean_squared_error,
        rmse=lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
    )

    # save evaluation outputs
    model_outputs_path = output_dir / 'model_outputs.pkl'
    with open(model_outputs_path, 'wb') as f:
        pickle.dump(model_outputs, f)

    wrong_predictions_path = output_dir / 'wrong_preds.pkl'
    with open(wrong_predictions_path, 'wb') as f:
        pickle.dump(wrong_predictions, f)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july')
    argparser.add_argument('--doms', nargs='*', default=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'])
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    args = argparser.parse_args()

    for dom in args.doms:
        test_pkl = PATHS.getpath(args.datapath) / f"clf_levels_{dom}_sents/test.pkl"
        test_dom_output_pkl = PATHS.getpath(args.datapath) / f"clf_levels_{dom}_sents/test_dom_output.pkl"
        model_name = PATHS.getpath(args.modelpath) / f"levels_{dom.lower()}_sents"
        test_output_dir = model_name / 'eval_test'
        test_dom_output_dir = model_name / 'eval_test_dom_output'

        print(f"Evaluating {model_name} on test.pkl")
        evaluate(
            test_pkl,
            args.model_type,
            str(model_name),
            test_output_dir,
        )

        print(f"Evaluating {model_name} on test_dom_output.pkl")
        evaluate(
            test_dom_output_pkl,
            args.model_type,
            str(model_name),
            test_dom_output_dir,
        )