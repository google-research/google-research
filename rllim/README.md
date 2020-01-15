# Codebase for "Reinforcement Learning based Locally Interpretable Modeling"

Authors: Jinsung Yoon, Sercan O. Arik, Tomas Pfister

Paper: Jinsung Yoon, Sercan O Arik, Tomas Pfister, "RL-LIM: Reinforcement
Learning-based Locally Interpretable Modeling", arXiv preprint arXiv:1909.12367
(2019). https://arxiv.org/abs/1909.12367

This directory contains implementations of Reinforcement Learning based Locally
Interpretable Modeling (RL-LIM) on three synthetic datasets and real datasets.

To run the pipeline for training and evaluation on real datasets, simply run
python3 -m main_rllim_on_real_data.py or take a look at the following
jupyter-notebook file - main_rllim_on_real_data.ipynb.

To run the pipeline for training and evaluation on synthetic datasets, simply
run python3 -m main_local_dynamics_recovery.py or take a look at the following
jupyter-notebook file - main_local_dynamics_recovery.ipynb.

## Experiment settings on experiments with real datasets

-   Loads facebook comment volume datasets (users can easily replace datasets).
-   Trains black-box model and generate auxiliary datasets.
-   Trains RL-LIM on auxiliary datasets and generate interpretable predictions
    and instance-wise explanations
-   Evaluates overall performance: check whether interpretable predictions are
    similar with the ground truth labels in terms of MAE.
-   Evaluates fidelity performance: check whether interpretable predictions are
    similar with black-box model predictions in terms of R2 score and MAE.
-   Reports the sample instance-wise explanations

### Command inputs:

-   problem: 'regression'
-   normalization: 'minmax'
-   train_rate: 0.9
-   probe_rate: 0.1
-   seed: 0
-   hyper_lambda: main hyper-parameter of RL-LIM (lambda) (1.0)
-   hidden_dim: hidden state dimensions (100)
-   iterations: number of RL iterations (2000)
-   num_layers: number of layers (5)
-   batch_size: the number of mini-batch samples for RL (5000)
-   batch_size_inner: the number of mini-batch samples for inner iterations (10)
-   n_exp: the number of sasmple explanations (5)
-   checkpoint_file_name: file name for saving and loading the trained model
    (./tmp/model.ckpt)

### Example command

```shell
$ python3 main_rllim_on_real_data.py --problem regression \
--train_rate 0.9 --probe_rate 0.1 --seed 0 --hyper_lambda 1.0 --hidden_dim 100 \
--iterations 2000 --num_layers 5 --batch_size 5000 --batch_size_inner 10 \
--n_exp 5 --checkpoint_file_name ./tmp/model.ckpt
```

### Outputs

-   Overall performance in terms of MAE
-   Fidelity performances in terms of R2 score and MAE
-   Sample instance-wise explanations

## Experiment setting on experiments with synthetic datasets

-   Generates 3 synthetic datasets with different characteristics.
-   Trains RL-LIM on training datasets and generate interpretable predictions
    and instance-wise explanations (local dynamics recovery)
-   Evaluates local dynamics recovery: check whether RL-LIM can recover the
    ground truth local dynamics in terms of Absolute Weight Differences (AWD).
-   Evaluates fidelity performance: check whether interpretable predictions are
    similar with black-box model predictions in terms of MAE.

### Command inputs:

-   data_name: 'Syn1' or 'Syn2' or 'Syn3'
-   train_no: 1000
-   probe_no: 100
-   test_no: 1000
-   dim_no: 11
-   seed: 0
-   hyper_lambda: main hyper-parameter of RL-LIM (lambda) (1.0)
-   hidden_dim: hidden state dimensions (100)
-   iterations: number of RL iterations (2000)
-   num_layers: number of layers (5)
-   batch_size: the number of mini-batch samples for RL (900)
-   batch_size_inner: the number of mini-batch samples for inner iterations (10)
-   checkpoint_file_name: file name for saving and loading the trained model
    (./tmp/model.ckpt)

### Example command

```shell
$ python3 main_local_dynamics_recovery.py --data_name Syn1 --train_no 1000 \
--probe_no 100 --test_no 1000 --dim_no 11 --seed 0 --hyperparam 1.0 \
--hidden_dim 100 --iterations 2000 --num_layers 5 --batch_size 900 \
--batch_size_inner 10 --checkpoint_file_name ./tmp/model.ckpt
```

### Outputs

-   Local dynamics recovery performance in terms of absolute weight differences
    (AWD) between ground truth local dynamics and estimated local dynamics by
    RL-LIM.
-   Fidelity performance in terms of MAE
-   Plots AWD and MAE with respect to distance from the boundary where the local
    dynamics change (in percentile)
