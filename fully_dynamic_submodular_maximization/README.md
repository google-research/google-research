# Dynamic submodular maximization

This is an efficient C++ implementation of the algorithms and baselines from
"Fully Dynamic Algorithm for Constrained Submodular Optimization".

The algorithms require oracle access to a submodular function.
In this code, such an oracle is implemented for graph coverage / influence maximization,
which is the setting for the experiments in the paper.
To implement your own submodular function oracle, inherit from SubmodularFunction.
Refer to comments in the code for more details.

To replicate the results given in the paper:
1. Edit the `main()` function at the end of `dynamic_submodular_main.cc` to choose the desired parameters:
  * dataset name (edit the line `GraphUtility f_graph("enron");` - see below for possible options)
  * cardinality constraint(s) k
  * type of experiment (sliding window, etc.)
  * number of repeats (for randomized algorithms)
  * which algorithms to run
  See the comments in `main()` for how to adjust the parameters.
2. Run `make` in this directory. This should produce a file `dynamic-submodular.exe`.
3. Run `dynamic-submodular.exe`.

As we use cross-platform-deterministic randomness and count only oracle calls rather than CPU time,
parameters of the system on which the experiments are run are irrelevant.

