# Tensor network contraction optimizer with simulated annealing

Tensor network (TN) optimization code and data for the simulation of random circuit sampling from [Phase transition in random circuit sampling](https://arxiv.org/abs/2304.11119).

## Description of folders
1. data/: circuit data and optimization results.
    - Files with the graph structure of the circuits (extension `.graph`).
    - Files with the sets of indices in the sets of indices of the TN that can be sliced together (extension `.groups`).

2. src/: C++ code with the optimizer.

3. scripts/: useful scripts.
    - `circuit_to_tn.py`: script to convert a Cirq circuit from the ensemble studied in this paper into a `.graph` and a `.groups` file. Accepted input file formats are json, qasm, and qsim.

## How to use it
1. Compile:
    - Move to the `src/` folder. From the main folder, run `cd src/`.
    - Run `make optimize -j`. The code has no dependencies.
You should find an executable named `optimize` in the `src/` folder.

2. Run in either one of the three modes:
    1) Optimize contraction ordering without memory constraints.
    ```
    ./optimize 1 <graph filename> <output filename> <random seed> <num SA steps> <initial SA temperature T0>  <final SA temperature T1>
    ```
    2) Optimize contraction ordering with width constraint, grouped slices, and sparse outputs.
    ```
    ./optimize 2 <graph filename> <groups filename> <width> <num sparse configurations> <output filename> <random seed> <num SA steps> <initial SA temperature T0>  <final SA temperature T1>
    ```
    3) Optimize contraction ordering with full memory constraint, grouped slices, and sparse outputs.
    ```
    ./optimize 3 <graph filename> <groups filename> <log2(memory footprint (num scalars))> <num sparse configurations> <output filename> <random seed> <num SA steps> <initial SA temperature T0>  <final SA temperature T1>
    ```
For help, run `./optimize -h` or `./optimize --help`.

### Example
From the `src/` folder, run:
    1) `./optimize 1 ../data/google_n53_m20.graph ../data/google_n53_m20.out 1 1000000 1e5 1e-5`
    2) `./optimize 2 ../data/google_n53_m20.graph ../data/google_n53_m20.groups ../data/google_n53_m20.out 1 33 1 100000 1e5 1e-5`
    3) `./optimize 3 ../data/google_n53_m20.graph ../data/google_n53_m20.groups ../data/google_n53_m20.out 1 35 1 100000 1e5 1e-5`
All three examples run with simulated annealing over 100000 steps.

The code is not parallelize, so we recommend setting `export OMP_NUM_THREADS=1` and running as many simulated annealing walkers as cores available. One should take the best result among all walkers. We also recommend running the code for a sequence of increasingly larger simulated annealing number of steps, in order to see convergence in the final answer. A typical number of steps might be one or a few million.

## Description of `.graph` and `.groups` file formats
The tensor network is specified in the `.graph` file. The tensor network has $`N`$ tensors and $`E`$ edges or indices. This file consists of as many rows as edges in the network. Each row consists of a few integers: the first one specifies the bond dimension on the index, while the subsequent integers specify the tensors this index is connected to. Tensors have to be labelled with consecutive integers from 0 to $`N-1`$. Label $`-1`$ is reserved to the *environment*, i.e., it is used for indices that are external to the network or open. While this format allows for hyperindices to be specified, the optimizer cannot handle hyperindices at the moment.

Some tensors allow for simulataneous slicing of two or more indices at a time. For that reason, it is useful to keep track of these groups or sets of indices. These are specified in the `.groups` file. Each row of this file is a *group*. In it, the indices that can be sliced together are specified. Indices are labelled after their position in the `.graph` file, i.e., with consecutive integers from 0 to $`E-1`$. Note that groups consisting only of a single index *have* to be specified in this file. Only groups (either consisting of a single index or more) appearing in this file are considered as potential slices.
