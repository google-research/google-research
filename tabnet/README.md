# Codebase for "TabNet: Attentive Interpretable Tabular Learning"

Authors: Sercan O. Arik, Tomas Pfister

Paper: https://arxiv.org/abs/1908.07442

This directory contains an example implementation of TabNet on the Forest Covertype dataset (https://archive.ics.uci.edu/ml/datasets/covertype). 

First, run `python -m download_prepare_covertype.py` to download and prepare the Forest Covertype dataset.
This command creates train.csv, val.csv and test.csv files under the "data/" directory.

To run the pipeline for training and evaluation, simply use `python -m experiment_covertype.py`.
For debugging in a low-resource environment, you can use `python -m test_experiment_covertype.py`.

To modify the experiment to other tabular datasets:
- Substitute the train.csv, val.csv, and test.csv files under "data/" directory,
- Modify the data_helper function with the numerical and categorical features of the new dataset,
- Reoptimize the TabNet hyperparameters for the new dataset.
