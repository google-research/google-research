# Constrained Prediction of Linguistic Typological Features

This directory contains the code developed for [SIGTYP 2020 Shared
Task](https://sigtyp.github.io/st2020.html) which involves prediction of
typological properties of languages given a handful of observed features. The
typological features are taken from the World Atlas of Language Structures
([WALS](https://wals.info/)).

## Dependencies

Please check [requirements.txt](requirements.txt) for a list of basic
dependencies for most of the tools in this directory.

## Usage Scenarios

In the following steps we assume all the commands are run from the current
directory where all the python code resides.

### Initial setup

The original shared task data is distributed here
[here](https://github.com/sigtyp/ST2020). Download it to some local directory:

```shell
WORK_DIR=/tmp/workspace
mkdir -p ${WORK_DIR}/sigtyp
git clone https://github.com/sigtyp/ST2020 ${WORK_DIR}/sigtyp
```

### Preprocessing the data

The pipeline internally uses a different csv format from the original
distribution. In particular, each WALS feature has its own dedicated column.
To generate data in the internal format please run:

```shell
mkdir ${WORK_DIR}/internal_data
python3 sigtyp_reader_main.py \
  --sigtyp_dir ${WORK_DIR}/sigtyp/data \
  --output_dir ${WORK_DIR}/internal_data
```

The above command will generate several files in `${WORK_DIR}/internal_data`
data directory:

*  The csv files containing various combinations of training, development and
   test data.
*  The compressed dictionaries in JSON format that contain miscellaneous
   WALS feature information.

### Preparing training and evaluation data

Compute the feature associations (such as implicational universals):

```shell
mkdir -p ${WORK_DIR}/associations/train
python3 compute_associations_main.py \
  --training_data ${WORK_DIR}/internal_data/train.csv \
  --dev_data ${WORK_DIR}/internal_data/dev.csv \
  --association_dir ${WORK_DIR}/associations/train
```

The above will generate several feature association files under
`${WORK_DIR}/associations/train`:

*  `raw_proportions_by_family.tsv`: MLE estimates for language families.
*  `raw_proportions_by_genus.tsv`: MLE estimates for language genera.
*  `raw_proportions_by_neighborhood.tsv`: MLE estimates based on geographic
   neighborhood.
*  `implicational_universals.tsv`: Implicational universals.

### Training and evaluating the models

#### Evaluating

Following will evaluate the random forest models trained using the training data
on the development set:

```shall
python3 evaluate_main.py \
  --sigtyp_dir ${WORK_DIR}/sigtyp/data \
  --training_data_dir ${WORK_DIR}/internal_data \
  --train_set_name train --test_set_name dev \
  --association_dir ${WORK_DIR}/associations/train \
  --algorithm NemoModel \
  --num_workers 10 \
  --force_classifier RandomForest
```

Alternatively, it is possible to evaluate individual features or groups of
features as follows:

```shell
python3 scikit_classifier_main.py \
  --training_data_file ${WORK_DIR}/internal_data/train.csv \
  --dev_data_file ${WORK_DIR}/internal_data/dev.csv \
  --data_info_file ${WORK_DIR}/internal_data/data_info_train_dev.json.gz \
  --association_dir ${WORK_DIR}/associations/train \
  --classifiers=SVM,DNN,LogisticRegression,AdaBoost,RandomForest \
  --target_feature Order_of_Subject,_Object_and_Verb \
  --nocatch_exceptions
```

The above will evaluate a bunch of classifiers trained for the
`Order_of_Subject,_Object_and_Verb` WALS feature.

#### Predicting

To predict unknown (missing) WALS features for the test data (which marks those
by `?`), first generate the associations for the combined training and
development set:

```shell
mkdir -p ${WORK_DIR}/associations/train_dev
python3 compute_associations_main.py \
  --training_data ${WORK_DIR}/internal_data/train_dev.csv \
  --dev_data ${WORK_DIR}/internal_data/test_blinded.csv \
  --association_dir ${WORK_DIR}/associations/train_dev
```

Then run prediction by enabling the prediction mode of
[evaluate_main.py](evaluate_main.py) as follows:

```shell
python3 evaluate_main.py \
  --sigtyp_dir ${WORK_DIR}/sigtyp/data \
  --training_data_dir ${WORK_DIR}/internal_data \
  --train_set_name train_dev \
  --test_set_name test_blinded \
  --association_dir ${WORK_DIR}/associations/train_dev \
  --algorithm NemoModel \
  --num_workers 10 \
  --force_classifier RandomForest \
  --prediction_mode \
  --output_sigtyp_predictions_file ${WORK_DIR}/test_results.csv
```

This will fill in the missing WALS features in the original SIGTYP 2020 csv
format in `${WORK_DIR}/test_results.csv`.
