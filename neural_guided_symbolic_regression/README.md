# Neural-Guided Symbolic Regression with Asymptotic Constraints

This directory contains reference code for paper
"Neural-Guided Symbolic Regression with Asymptotic Constraints"

https://arxiv.org/abs/1901.07714

## Dataset

`grammar/univariate_one_constant_grammar.txt` contains grammar production rules
to generate univariate expressions.

Folder `data` contains datasets of rational expressions used in the paper.
Each CSV contains four columns:
`expression_string`,
`simplified_expression_string`,
`leading_at_0`,
`leading_at_inf`

We define the complexity of an expression as

M = |leading power at 0| + |leading power at inf|


### `train.csv`

28837 expressions with M <= 4. This dataset is used for training the neural
network model and generation of empirical distribution for baseline models.
`train.tfrecords` contains data in `train.csv`.

### `eval.csv`

4095 expressions with M <= 4. This dataset is used for validation in neural
network training.
`eval.tfrecords` contains data in `eval.csv`.

### `holdout_m_leq_4.csv`

2050 expressions with M <= 4.

### `holdout_m_5.csv`

1000 expressions with M = 5.

### `holdout_m_6.csv`

1200 expressions with M = 6.


`holdout_m_leq_4.csv`, `holdout_m_5.csv` and `holdout_m_6.csv` are used to
evaluate different symbolic regression methods.

## Metrics

The code to evaluate success, L1-distance, syntactic novelty and
semantic novelty:

`utils/expression_generalization_metrics.py`

## Expression Generating Neural Network and Baseline Models

The reference code to train the neural network is in folder `models`.

Train neural network (NN):

```
python -m neural_guided_symbolic_regression.models.run_partial_sequence_model \
--hparams=$(pwd)/neural_guided_symbolic_regression/models/config/neural_network.json \
--is_chief \
--model_dir=/tmp/neural_guided_symbolic_regression/example_run
```

Train neural network no condition (NNNC):

```
python -m neural_guided_symbolic_regression.models.run_partial_sequence_model \
--hparams=$(pwd)/neural_guided_symbolic_regression/models/config/neural_network_no_condition.json \
--is_chief \
--model_dir=/tmp/neural_guided_symbolic_regression/example_run
```

The code to generate empirical distribution baseline models:

`utils/generate_empirical_distribution_df.py`

## Symbolic Regression Methods

### Monte Carlo Tree Search

(Neural-Guided) Monte Carlo Tree Search is implemented in folder `mcts`.
`models/mcts.py` defines policy and reward for the symbolic regression tasks
in the paper.

### Evolutionary Algorithm

Evolutionary algorithm is implemented in `models/evolutionary.py`.



