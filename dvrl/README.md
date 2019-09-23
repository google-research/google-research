# Codebase for "Data Valuation using Reinforcement Learning"

Authors: Jinsung Yoon, Sercan O. Arik, Tomas Pfister

Paper: https://openreview.net/forum?id=BJxEciVhLH

This directory contains an example implementation of Data Valuation using
Reinforcement Learning (DVRL) on the UCI Adult Income dataset
(https://archive.ics.uci.edu/ml/datasets/Adult) and Blog Feedback dataset
(https://archive.ics.uci.edu/ml/datasets/BlogFeedback) with assuming a LightGBM
as the black-box model.

With this directory, we can replicate partial results in the paper described in
Fig. 2, 3, 4, 8, and 9.

To run the pipeline for training and evaluation, simply run python -m
experiment_main.py.

## Experiment setting

-   Add 0% or 20% of noise on labels
-   Check whether DVRL can discover the corrupted samples well
-   Check the prediction performances of predictive model (LightGBM
    (https://lightgbm.readthedocs.io/en/latest/Python-Intro.html) in this
    implementation) after removing most/least valuable samples determined by
    DVRL

### Command inputs:

-   data_name: 'adult' or 'blog'
-   train_no: 1000
-   valid_no: 400
-   noise_rate: 0.0 or 0.2
-   hidden_dim: hidden state dimensions (100)
-   iterations: number of RL iterations (2000)
-   layer_number: number of layers (5)
-   batch_size: the number of mini-batch samples for RL (2000)

### Example command

```shell
$ python3 experiment_main.py --data_name adult --train_no 1000 --valid_no 400 \
--noise_rate 0.2 --hidden_dim 100 --iterations 2000 \
--layer_number 5 --batch_size 2000
```

### Outputs

-   Noisy sample discovery performance (only if noise_rate > 0)
-   Performance after removing least valuable samples
-   Performance after removing most valuable samples
