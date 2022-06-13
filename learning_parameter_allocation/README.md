# Flexible Multi-task Networks by Learning Parameter Allocation

This package includes the code for reproducing the experiments from the paper:
"Flexible Multi-task Networks by Learning Parameter Allocation".
https://arxiv.org/abs/1910.04915

## Setup

You should have python3.6 and python3.6-dev installed on your machine:

```
apt-get install python3.6 python3.6-dev
```

To create a virtual environment with the necessary dependencies, run:

```
source ./learning_parameter_allocation/run.sh
```

## Running experiments

All experiments save logs and summaries under the logdir given as a commandline flag.
When running several experiments in a row, make sure that this logdir is emptied between the runs.

### MNIST (Section 5.1)

For MNIST, the training process evaluates the train/test accuracy at the end of training.

To run the training (four variants depending on the method):

```
python -m learning_parameter_allocation.mnist.mnist_train \
  --method=no_sharing
```

```
python -m learning_parameter_allocation.mnist.mnist_train \
  --method=shared_bottom
```

```
python -m learning_parameter_allocation.mnist.mnist_train \
  --method=gumbel_matrix
```

```
python -m learning_parameter_allocation.mnist.mnist_train \
  --method=gumbel_matrix --budget=0.75 --budget_penalty=1.0
```

### Omniglot (Section 5.2)

For Omniglot, the training process also evaluates the train/validation/test accuracies at the end of training.

However, a separate validation process can be used to run evals for all checkpoints, and pick the best checkpoint based on validation accuracy.

To run the training (two variants depending on the method):

```
python -m learning_parameter_allocation.omniglot.omniglot_train \
  --method=shared_bottom
```

```
python -m learning_parameter_allocation.omniglot.omniglot_train \
  --method=gumbel_matrix
```

To run the additional evaluation process:

```
python -m learning_parameter_allocation.omniglot.omniglot_eval \
  --method=shared_bottom
```

```
python -m learning_parameter_allocation.omniglot.omniglot_eval \
  --method=gumbel_matrix
```


### Three task clusters (Section 5.3)
To run the three task clusters experiment:

```
python -m learning_parameter_allocation.clusters.clusters_train
```

To display the learned patterns and pairwise task similarities:

```
python -m learning_parameter_allocation.clusters.plot \
  --dir=path/to/summary/file
```
