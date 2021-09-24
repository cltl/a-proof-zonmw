Prepare datasets for machine learning
====================================
# Data for domains classifier
The script [data_prep_domains.py](data_prep_domains.py) processes parsed annotations pkl's that are created by the [parse_annotations.py](../data_process_from_inception/parse_annotations.py) script and prepares train, dev and test datasets for domains classification.

For more details, see the docstring in the script.

# Data for levels classifiers
The script [data_prep_levels.py](data_prep_levels.py) processes parsed annotations pkl's that are created by the [parse_annotations.py](../data_process_from_inception/parse_annotations.py) script and prepares train, dev and test datasets for levels classification.

You should run this script after you have predictions from the domains classifier, for two reasons:

- The data split is based on the split used for the domains classifier, e.g. the notes that were in the train set of the domains classifier are in the train set for the levels classifiers as well.
- Two evaluation sets are created for each domain: one that contains the sentences that have gold levels labels, and one that contains the sentences that were assigned a domain label by the domains classifier and do not necessarily have a gold label. This is necessary for evaluation of the impact of 'background' sentences on the note-level scores. It can be configured in the script whether the evaluation set is the dev or the test set.

For more details, see the docstring in the script.
