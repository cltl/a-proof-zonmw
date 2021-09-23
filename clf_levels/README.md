Levels regression models
=======================
# Description
Fine-tuning a pretrained language model for regression models predicting functioning levels per domain.

For more details about regression with Simple Transformers, see the [documentation](https://simpletransformers.ai/docs/regression/).

# General configuration options
For all the scripts in this repo, the default is that the training/evaluation/prediction is done for all 9 domains and the classification unit is a sentence.

This can be configured with the `--doms` and `--clas_unit` arguments, for example:
```
$ python predict.py --doms ADM, ENR --clas_unit note
```

# Training
You can fine-tune a new model with the [train_model.py](train_model.py) script.
## Configuring model args
- The model args are costumized in the [config.json](config.json) file.
- For all available args, see the Simple Transformers [documentation](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model).

## Pretrained language models
The language model that is used for fine-tuning can be either locally stored or a model from [Hugging Face](https://huggingface.co/models). The type and name of the model are passed as arguments to the [train_model.py](train_model.py) script. For example, to use the pretrained [RobBERT](https://huggingface.co/pdelobelle/robbert-v2-dutch-base) from Hugging Face:
```
$ python train_model.py --model_type roberta --model_name pdelobelle/robbert-v2-dutch-base --hf
```
Alternatively, you can edit the default values of the parameters in the [script itself](train_model.py).

# Evaluating
The [evaluate_model.py](evaluate_model.py) script calculates evaluation loss, MAE (mean absolute error), MSE (mean squared error) and RMSE (root mean squared error). In addition, it stores model outputs and wrong predictions.

In the current project, in addition to the sentence-level metrics calculated by the script, we are interested in aggregated note-level metrics. For this purpose, the [predict.py](predict.py) script is used to generate predictions on the evaluation set, and the metrics of interest are calculated with the [clf_levels_eval_note_agg.py](../ml_evaluation/clf_levels_eval_note_agg.py) script.

# Predicting
The [predict.py](predict.py) script takes as input a pickled DataFrame with a 'text' column and adds a column with predictions per row.
