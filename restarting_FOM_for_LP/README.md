# Restarts for primal-dual methods

## Introduction

This is the code accompanying the paper Applegate, David, et al. "Faster
first-order primal-dual methods for linear programming using restarts and
sharpness." https://arxiv.org/pdf/2105.12715.pdf. It includes a simple
implementation of PDHG, extragradient and scripts to run all the numerical
experiments appearing in the paper.

## Running

Use the following scripts to run the experiments. The scripts use Julia 1.6.5.
All commands below assume that the current directory is the working directory.

A one-time step is required to set up the necessary packages on the local
machine:

```shell
$ julia --project -e 'import Pkg; Pkg.instantiate()'
```

This needs to be run again only if the dependencies change.

To test the code:

```shell
$ julia --project tests/tests.jl
```

To download the instances:

```shell
$ ./scripts/collect_qap_lp_instances.sh [temporary_directory] [output_directory]
```

Note that this code may take a couple of days to run with problem_name=all:

```shell
$ julia --project scripts/run_problems.jl [test_problem_folder] [results_directory] [method (PDHG or extragradient)] [problem_name (all, qap10, qap15, nug08-3rd, nug20)]
```

To plot the results:

```shell
$ julia --project scripts/plot_results.jl [results_directory]
```

To reproduce the 1D bilinear plots:

```shell
$ julia --project scripts/run_and_plot_bilinear_example.jl [results_directory] [method (PDHG or extragradient)]
```

Run the following command to auto-format all Julia code in this directory before
submitting changes:

```shell
$ julia --project=formatter -e 'using JuliaFormatter; format(".")'
```
