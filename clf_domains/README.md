Domain classification
=====================
# Description
Fine-tuning a pretrained language model for the task of multi-label classification of WHO-ICF domains.

For more details about multi-label classification with Simple Transformers, see the [tutorial](https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5) and the [documentation](https://simpletransformers.ai/docs/multi-label-classification/).

# Training
You can fine-tune a new model with the [train_model.py](train_model.py) script.
## Configuring model args
- The model args are costumized in the [config.json](config.json) file.
- For all available args, see the Simple Transformers [documentation](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model).
- In addition to the general args (listed in the link above), the multi-label classification model has an option to configure the `threshold` at which a given label flips from 0 to 1 when predicting; see [here](https://simpletransformers.ai/docs/classification-models/#configuring-a-multi-label-classification-model) for details.

## Pretrained language models
The language model that is used for fine-tuning can be either locally stored or a model from [Hugging Face](https://huggingface.co/models). The type and name of the model are passed as arguments to the [train_model.py](train_model.py) script. For example, to use the pretrained [RobBERT](https://huggingface.co/pdelobelle/robbert-v2-dutch-base) from Hugging Face:
```
$ python train_model.py --model_type roberta --model_name pdelobelle/robbert-v2-dutch-base --hf
```
Alternatively, you can edit the default values of the parameters in the [script itself](train_model.py).

# Evaluating
The [evaluate_model.py](evaluate_model.py) script calculates evaluation loss and LRAP (Label Ranking Average Precision). It also saves the model outputs and the wrong predictions.

In the current project, we were interested in calculating precision, recall and F1-score per domain, so the above outputs were not the most convenient. Instead, the [predict.py](predict.py) script was used to generate predictions on the evaluation set, and the metrics of interest were calculated in the [clf_domains_eval.ipynb](../ml_evaluation/clf_domains_eval.ipynb) notebook.

# Predicting
The [predict.py](predict.py) script takes as input a pickled DataFrame with a 'text' column and adds a column with predictions per row. The type and name of the locally stored model that is used to predict is passed as arguments in the command line (alternatively, edit the default values in the script).

# Hyper-parameter optimization
The [wandb_sweep.py](wandb_sweep.py) script performs a sweep for hyperparameters optimization, using Simple Transformers and W&B Sweeps.

The sweep is configured in a dictionary in the [config.json](config.json) file, which should specify the search strategy, the metric to be optimized, and the hyperparameters (and their possible values).

For more details, see [here](https://simpletransformers.ai/docs/tips-and-tricks/#hyperparameter-optimization).
