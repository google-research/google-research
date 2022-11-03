# Restarts for primal-dual methods

## Introduction

This is the code accompanying the paper Applegate, David, et al. "Faster
first-order primal-dual methods for linear programming using restarts and
sharpness." https://arxiv.org/pdf/2105.12715.pdf. It includes a simple
implementation of PDHG, extragradient, and ADMM with scripts to run all the numerical
experiments appearing in the paper.

## Setup

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
$ ./scripts/collect_qap_lp_instances.sh \
  [temporary_directory] [directory_for_problem_instances]
```

## To run all the results

```shell
$ ./scripts/run_all.sh \
  [directory_for_problem_instances] \
  [output_directory]
```

## Running individual methods or instances

To run all the results* for an individual method
(note that this code may take a couple of days to run with problem_name=all):

*To conserve compute this doesn't run NoRestarts() for 500,000 iterations
which is necessary to fully replicate the tables. To do this, see [Running one configuration](#running-one-configuration) below.

```shell
$ julia --project scripts/run_problems.jl \
  [directory_for_problem_instances] \
  [results_directory] \
  [method (ADMM, PDHG or extragradient)] \
  [problem_name (all, qap10, qap15, nug08-3rd, nug20)]
```

## Plotting results

```shell
$ julia --project \
  scripts/plot_results.jl \
  [results_directory]
```

To reproduce the 1D bilinear plots:

```shell
$ julia --project \
  scripts/run_and_plot_bilinear_example.jl \
  [results_directory] \
  [method (PDHG or extragradient)]
```


## Running one configuration {#running-one-configuration}

```shell
$ julia --project scripts/run_configuration.jl \
  [directory_for_problem_instances] [results_csv_file] \
  [method (ADMM, PDHG or extragradient)] \
  [problem_name (qap10, qap15, nug08-3rd, nug20)] \
  [restart_scheme (no_restarts, fixed_frequency or adaptive)] \
  [restart_length] [always_reset_to_average (yes or no)] \
  [iteration_limit] [kkt_tolerance]
```

For example:

```shell
$ julia --project scripts/run_configuration.jl \
  [directory_for_problem_instances] \
  [results_csv_file] \
  ADMM \
  qap10 \
  no_restarts \
  10 \
  no \
  500000 \
  1e-6
```

This command can be useful if you just want to quickly verify a particular
configuration.

## Auto Formatter

A one-time step is required to use the auto-formatter:

```shell
$ julia --project=formatter -e 'import Pkg; Pkg.instantiate()'
```


Run the following command to auto-format all Julia code in this directory before
submitting changes:

```shell
$ julia --project=formatter -e 'using JuliaFormatter; format(".")'
```
