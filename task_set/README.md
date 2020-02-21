# TaskSet: A dataset of tasks for evaluating and training optimizers
This directory contains a variety of optimization problems for use in evaluating
and meta-training learned optimizers. It is decribed in
"Using a thousand optimization tasks to learn hyperparameter search strategies"
[arxiv](link_to_paper).

The problems are implemented as tensorflow 1.x style models mostly using Sonnet.

# Usage to train models
In addition to model definition, we also provide training scripts.

`python3 -m task_set.train_inner --optimizer_name="adam4p_wide_grid_seed107" --task_name="mlp_family_seed117" --output_directory="/tmp/root_data_dir"`

# Requirements:
As of now, we only support tensorflow version 1.0 (e.g. tensorflow-1.15) and
1.x sonnet. See requirements.txt for full versions required.
