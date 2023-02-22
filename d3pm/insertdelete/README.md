This directory contains code for the ICML 2021 INNF+ workshop paper "Beyond In-Place
Corruption: Insertion and Deletion In Denoising Probabilistic Models".

```
@inproceedings{johnson2021beyond,
author    = {Daniel D. Johnson and
             Jacob Austin and
             Rianne van den Berg and
             Daniel Tarlow},
title     = {Beyond In-Place Corruption: Insertion and Deletion In Denoising Probabilistic Models},
booktitle = {ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models},
year      = {2021}
}
```

## Interactive guide notebook

You can explore the insertion-deletion forward process using the interactive guide available here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook_demo]

[notebook_demo]: https://colab.research.google.com/github/google-research/google-research/blob/master/d3pm/insertdelete/Insertion_and_Deletion_Forward_Process_Guide.ipynb

## Code organization

The code is organized as follows:

- Main logic
  - `forward_process.py` contains the code for building, sampling from, and running inference, based on Probabilistic Finite State Transducers.
  - `transition_operator.py` defines classes that handle the inner Markov transition matrix of the forward process.
  - `schedules.py` contains helper classes and functions for building diffusion schedules, which determine the speed of mixing of the forward process.
  - `training_setup.py` contains the top-level code for building losses and schedules used for training. (Due to dependencies on non-open-source libraries, this code is released in terms of a black-box model prediction function, and does not currently include logic for constructing and training the model itself.)

- Utility modules
  - `distributions.py` contains implementations of a variety of relevant probability distributions used by the forward process.
  - `dynamic_programs.py` contains JAX logic for running dynamic programming computations, which can be used to do more expensive inference steps (although these were not needed for model training).
  - `math_util.py` contains miscellaneous math-related utilities.
  - `util.py` contains other miscellaneous utilities.

