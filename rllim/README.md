# Codebase for "Reinforcement Learning based Locally Interpretable Models"

Authors: Jinsung Yoon, Sercan O. Arik, Tomas Pfister

Paper: https://openreview.net/forum?id=Bkehv54h8B

This directory contains an example implementation of Reinforcement Learning
based Locally Interpretable Models (RL-LIM) on three synthetic datasets.

With this directory, we can replicate the results in the paper described in Fig.
2, 3, and 4.

To run the pipeline for training and evaluation, simply run python -m
experiment_main.py.

## Experiment setting

-   Generate 3 synthetic datasets with different characteristics
-   Using RL-LIM to recover the local dynamics
-   Check whether RL-LIM can recover the ground truth local dynamics in terms of
    Absolute Weight Differences (AWD)

### Command inputs:

-   data_name: 'Syn1' or 'Syn2' or 'Syn3'
-   data_no: 1000
-   seed: 0
-   hyperparam: 1.0 (lambda)
-   hidden_dim: hidden state dimensions (100)
-   iterations: number of RL iterations (5000)
-   layer_number: number of layers (5)
-   batch_size: the number of mini-batch samples for RL (900)
-   batch_size_small: the number of mini-batch samples for inner iterations (10)

### Example command

```shell
$ python3 experiment_main.py --data_name Syn1 --data_no 1000 --seed 0 \
--hyperparam 1.0 --hidden_dim 100 --iterations 2000 \
--layer_number 5 --batch_size 900 --batch_size_small 10
```

### Outputs

-   Absolute Weight Differences (AWD) of RL-LIM for 10 divisions divided by the
    distance from the decision boundary
