Example code for running Metric-Optimized Example Weights (MOEW) algorithm.

Paper:
https://arxiv.org/abs/1805.10582

## MNIST Dataset:

Example run:

`python -m mnist`

## Wine Dataset:

Download the dataset from:
https://www.kaggle.com/dbahri/wine-ratings

Example run:

```shell
python -m wine \
  --training_data_path=train.csv \
  --testing_data_path=test.csv \
  --validation_data_path=validation.csv
```

## Communities and Crime Dataset:

Download the dataset from:
http://archive.ics.uci.edu/ml/datasets/communities+and+crime

```shell
python -m crime \
  --training_data_path=train.csv \
  --testing_data_path=test.csv \
  --validation_data_path=validation.csv
```
