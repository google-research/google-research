# Learning Graph Structure With A Finite-State Automaton Layer

This directory contains the implementation of the Graph Finite-State Automaton
layer described in

["Learning Graph Structure With A Finite-State Automaton Layer"](https://arxiv.org/abs/2007.04929)

Daniel D. Johnson, Hugo Larochelle, Daniel Tarlow (2020).

If you use the code, models, or data in this repository, please cite the following paper:
```
@inproceedings{gfsa,
author    = {Daniel D. Johnson and
             Hugo Larochelle and
             Daniel Tarlow},
title     = {Learning graph structure with a finite-state automaton layer},
booktitle = {Advances in Neural Information Processing Systems},
year      = {2020}
}
```

## Interactive demo notebooks

Want to see the GFSA layer in action? A good starting point is the interactive
demo notebook, which shows how to train the GFSA layer to do a simple static
analysis of Python code:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook_demo]

You may also be interested in the [new task guide notebook][notebook_new_task_guide],
which describes how to use the GFSA layer for new types of graphs and graph-based
MDPs.

[notebook_demo]: https://colab.research.google.com/github/google-research/google-research/blob/master/gfsa/notebooks/demo_learning_static_analyses.ipynb
[notebook_new_task_guide]: https://colab.research.google.com/github/google-research/google-research/blob/master/gfsa/notebooks/guide_for_new_tasks.ipynb

## Setting up the environment

The code in this repository is written for Python 3.6. We recommend creating
a virtual environment and then installing the requirements in
`requirements.txt`. You may also want to configure your JAX installation for
GPU support; see the [JAX documentation](https://github.com/google/jax#installation)
for details.

## Structure of this repository

### Core implementation of the GFSA solver

- `graph_types.py` defines data structures for representing graph MDPs.
- `automaton_builder.py` is responsible for encoding graphs into tensors and
  computing the GFSA absorbing distribution.
- `automaton_sampling.py` implements the RL ablation of the GFSA layer.

### General utilities

- `jax_util.py` contains utilities for working with JAX and Flax.
- `linear_solvers.py` implements the Richardson iterative solver.
- `schema_util.py` defines helper functions for working with MDP families with
  a shared action and observation space.
- `sparse_operator.py` implements a sparse operator abstraction.

### Working with ASTs

- `generic_ast_graphs.py` defines a transformation from ASTs to MDPs.
- `py_ast_graphs.py` defines an AST for a simple subset of Python.
- `ast_spec_inference.py` can be used to construct an MDP family from a
  dataset of ASTs.

### Working with datasets

- `datasets/graph_bundle.py` defines data structures for working with graphs
  that are associated with encoded MDP.
- `datasets/graph_edge_util.py` implements helpers to construct MDPs based on
  graph edges.
- `datasets/data_loading.py` implements a pure-Python collection of dataset
  iterators.
- `datasets/padding_calibration.py` helps determine maximum example sizes that
  do not throw out too many examples.
- `datasets/mazes` defines MDPs and data-generation for the grid-world task.
- `datasets/random_python/top_down_refinement.py` implements a generalized
  AST generator based on a probabilistic context-free grammar.
- `datasets/random_python/python_numbers_control_flow.py` contains the specific
  generator used for the static analysis tasks.
- `datasets/var_misuse/example_definition.py` defines data structures for the
  variable-misuse task.

### Flax modules

The `model` subdirectory implements the GFSA layer, other graph architectures,
and combined models as Flax modules.

- `model/automaton_layer.py` contains the GFSA layer itself.
- `model/graph_layers.py` contains various graph architecture building blocks.
- `model/edge_supervision_models.py` assembles these blocks into models for the
  Python static analysis tasks.
- `model/end_to_end_stack.py` unifies the APIs of the building blocks so that
  they can be freely composed.
- `model/var_misuse_models.py` contains the implementation of the full models
  for the variable misuse tasks.
- `model/model_util.py` and `model/side_outputs.py` define some helper functions
  for Flax models.

### Training

- `training/configs` contains example gin-config configuration files for
  training a model.
- `training/simple_train.py` is the main entry point for training or evaluating
  a model on the three tasks described in the paper.
- `training/train_util.py` and `training/simple_runner.py` contain common logic
  for training between the three tasks.
- `training/train_edge_supervision_lib.py`, `training/train_maze_lib.py`, and
  `training/train_var_misuse_lib.py` contain the logic for each of the three
  tasks.
- `training/learning_rate_schedules.py` implements some simple learning rate
  schedules.
- `training/gin_util.py` defines a helper function for writing complex gin
  configurations.
