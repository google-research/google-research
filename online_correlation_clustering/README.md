# Online and Consistent Correlation Clustering

This code allows you to generate results from the ICML'22 paper: Online and
Consistent Correlation Clustering.

## This is a command to compile the code

g++ agreement_algo.h agreement_algo.cc graph_handler.cc graph_handler.h \\
main.cc pivot_algo.cc pivot_algo.h random_handler.cc random_handler.h \\
utils.cc utils.h -I/path/to/base/dir/ -O2 -o onlinecc

## The command to run the algorithm on input_file.

./onlinecc <input_file >output_file

## Parameter setting

The main parameters of the algorithm can be set in the main.cc file:

*   beta: the parameter beta of the agreement algorithm.
*   lambda: the parameter lambda of the agreement algorithm.
*   timestamped_input: if set to true, the arrival order is set to be with
    respect the increasing order of the ids of the nodes.

## Input description

The code expects one line for each '+' edge in the instance. The missing edges
are treated as '-' edges.

Input example for the graph with '+' edges (1,2), (1,3), (2,3): 1 2 1 3 2 3

## Output description

The output describes, for each algorithm that is considered, three values
following each update:

*   The cost of the solution produced by the algorithm, in terms of the
    correlation clustering objective.
*   The cummulative recourse of the algorithm, up until the last update.
*   The cummulative running time of the algorithm, up until the last update.

For more information, see the main.cc file.
