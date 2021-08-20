"""
Evaluate a fine-tuned regression model on a test set.
Save the following outputs:
- evaluation metrics (MSE, RMSE, MAE, eval_loss):
    saved in a `eval_results.txt` file in the model directory
- model outputs, wrong predictions:
    saved in the path indicated by the respective parameters
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
    model_outputs_path,
    wrong_predictions_path,
):
    """
    Evaluate a fine-tuned regression model on a test set.
    Save evaluation metrics in a `eval_results.txt` file in the model directory. The metrics include: MSE, RMSE, MAE and eval_loss.
    Save model outputs and wrong predictions in pickled files at `model_outputs_path` and `wrong_predictions_path`.

    Parameters
    ----------
    test_pkl: str
        path to pickled df with the test data, which must contain the columns 'text' and 'labels'; the labels are numeric values on a continuous scale
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        path to a directory containing model file
    model_outputs_path: str
        path to save the pickled model outputs
    wrong_predictions_path: str
        path to save the pickled wrong predictions

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
        mae=mean_absolute_error,
        mse=mean_squared_error,
        rmse=lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
    )

    # save evaluation outputs
    with open(model_outputs_path, 'wb') as f:
        pickle.dump(model_outputs, f)

    with open(wrong_predictions_path, 'wb') as f:
        pickle.dump(wrong_predictions, f)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data_expr_july')
    argparser.add_argument('--test_pkl', default='clf_levels_ADM_notes/test.pkl')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='models')
    argparser.add_argument('--model_name', default='levels_adm_notes')
    argparser.add_argument('--model_outputs', default='model_outputs_test.pkl')
    argparser.add_argument('--wrong_preds', default='wrong_preds_test.pkl')
    args = argparser.parse_args()

    test_pkl = PATHS.getpath(args.datapath) / args.test_pkl
    model_name = str(PATHS.getpath(args.modelpath) / args.model_name)
    model_outputs = PATHS.getpath(args.modelpath) / args.model_name / args.model_outputs
    wrong_predictions = PATHS.getpath(args.modelpath) / args.model_name / args.wrong_preds

    evaluate(
        test_pkl,
        args.model_type,
        model_name,
        model_outputs,
        wrong_predictions,
    )
