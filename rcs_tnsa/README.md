# Tensor network contraction optimizer with simulated annealing

Tensor network (TN) optimization code and data for the simulation of random circuit sampling from *Phase transitions in random circuit sampling* ([arXiv](https://arxiv.org/abs/2304.11119) and [Nature](https://www.nature.com/articles/s41586-024-07998-6)).

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

    1. Optimize contraction ordering without memory constraints.
    ```
    ./optimize 1 <graph filename> <output filename> <random seed> <num SA steps> <initial SA temperature T0>  <final SA temperature T1>
    ```
    2. Optimize contraction ordering with width constraint, grouped slices, and sparse outputs.
    ```
    ./optimize 2 <graph filename> <groups filename> <output filename> <random seed> <width> <num sparse configurations> <num SA steps> <initial SA temperature T0>  <final SA temperature T1>
    ```
    3. Optimize contraction ordering with full memory footprint constraint, grouped slices, and sparse outputs.
    ```
    ./optimize 3 <graph filename> <groups filename> <output filename> <random seed> <log2(memory footprint (num scalars))> <num sparse configurations> <num SA steps> <initial SA temperature T0> <final SA temperature T1>
    ```

For help, run `./optimize -h` or `./optimize --help`.

### Examples
From the `src/` folder, run:

1. `./optimize 1 ../data/google_n53_m20.graph ../data/google_n53_m20.out 1 100000 1e5 1e-5`
2. `./optimize 2 ../data/google_n53_m20.graph ../data/google_n53_m20.groups ../data/google_n53_m20.out 1 33 1000 100000 1e5 1e-5`
3. `./optimize 3 ../data/google_n53_m20.graph ../data/google_n53_m20.groups ../data/google_n53_m20.out 1 35 1000 100000 1e5 1e-5`

All three examples run with simulated annealing over 100000 steps. Mode `2` and `3` examples consider the computation of 1000 sparse outputs.

The code is not parallelized, so we recommend setting `export OMP_NUM_THREADS=1` and running as many independent simulated annealing walkers as cores available, each one with a different `random seed`. One should take the best result among all walkers. We also recommend running the code for a sequence of increasingly larger simulated annealing numbers of steps, in order to see convergence in the final answer. A typical target number of steps might be one to a few million.

## Description of `.graph` and `.groups` file formats
The tensor network is specified in the `.graph` file. The tensor network has `N` tensors and `E` edges or indices. This file consists of as many rows as edges in the network. Each row consists of a few integers: the first one specifies the bond dimension on the index, while the subsequent integers specify the tensors this index is connected to. Tensors have to be labelled with consecutive integers from 0 to `N-1`. Label `-1` is reserved to the *environment*, i.e., it is used for indices that are external to the network, i.e., open indices. While this format allows for hyperindices to be specified, the optimizer cannot handle hyperindices at the moment.

Some tensors allow for simultaneous slicing of two or more indices at a time. For that reason, it is useful to keep track of these groups or sets of indices. These are specified in the `.groups` file. Each row of this file is a *group*. In it, the indices that can be sliced together are specified. Indices are labelled after their position in the `.graph` file, i.e., with consecutive integers from 0 to `E-1`. Note that groups consisting only of a single index *have* to be specified in this file. Only groups (either consisting of a single index or more) appearing in this file are considered as potential slices.

## Output
The output of the code is a file with a description of the optimization mode,
other optimization parameters, and the resulting contraction cost and
contraction strategy. The contraction strategy consists of two parts:

1. **Contraction ordering**, under `Ordering`. This is provided as a list of all `E` index labels (integers from `0` to `E-1`) in the order they are contracted.

    *Note*: by convention, when an index between two tensors is contracted, it is implicit that *all* indices between these two tensors are contracted. This avoids creating inefficient self loops in the resulting intermediate tensor networks during contraction. Since the contraction ordering list contains all `E` indices, this implies that certain indices appearing later in the list will have to be ignored. Given that one avoids self loops in intermediate tensor networks there should only be `N-1` contractions in the whole procedure, while there are `E` indices in the tensor network. Therefore, one should expect to ignore `E-N+1` of the `E` indices at the time they appear in the contraction ordering list.

2. **Slices**, under `Sliced groups`. This field, which only appears in output from mode `2` and `3` runs, consists of a number of rows, each one containing a set of integers. Each row specifies a group or set of indices that are sliced simultaneously to the same value. These groups can contain one or more indices, and form a subset of all groups specified in the `.groups` input file.

### Tensor reuse, dynamic programming, or memoization
The contraction strategies provided for mode `2` and `3` runs imply a certain amount of dynamic programming or memoization. In practice, this means that some of the tensors computed in intermediate steps of the contraction are reused across different slice instances over the sliced groups of indices. This provides an improvement in the time complexity at the expense of an increased memory usage. Details on the tensor reuse strategy assumed and modelled by the optimizer can be found in **Appendix G** of the **Supplementary information** of the [paper](https://www.nature.com/articles/s41586-024-07998-6).

Similarly, mode `2` and `3` runs consider sparse outputs of a certain dimension passed as an input parameter. This avoids contracting the tensor network multiple times for different output configurations, without incurring in the large overhead that leaving all output indices open would imply. Refer also to the **Supplementary information** of the [paper](https://www.nature.com/articles/s41586-024-07998-6) for more details.
