# TaskSet: A dataset of tasks for evaluating and training optimizers
This directory contains a variety of optimization problems for use in evaluating
and meta-training learned optimizers. It is decribed in
"Using a thousand optimization tasks to learn hyperparameter search strategies"
[arxiv](https://arxiv.org/abs/2002.11887).

The problems are implemented as tensorflow 1.x style models mostly using Sonnet.

## Learning curves of trained models
As part of this repository we are releasing learning curves and corresponding
hyperparameters for roughly 29 million models. The data is stored in a cloud
bucket in npz format here: `gs://task_set_data/task_set_data/`.

For ease of analysis, we provide a sample [colab](https://colab.research.google.com/drive/1BYJzTx2MiJWbM4ydFoQvu2yon65banBj).

We hope this data can be used to gain insight into both optimizers, and probe notions of task similarity.
See our [paper](https://arxiv.org/abs/2002.11887) for examples of what can be done with data.

## Usage to train models
In addition to model definition, we also provide training scripts.

`python3 -m task_set.train_inner --optimizer_name="adam4p_wide_grid_seed107" --task_name="mlp_family_seed117" --output_directory="/tmp/root_data_dir"`


## Requirements:
As of now, we only support tensorflow version 1.0 (e.g. tensorflow-1.15) and
1.x sonnet. See requirements.txt for full versions required.
