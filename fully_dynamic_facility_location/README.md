# Efficient and Stable Fully Dynamic Facility Location

This code allows you to generate results from the NeurIPS'22 paper: Efficient
and Stable Fully Dynamic Facility.

# Location

## This is a command to compile the code

g++ gklx_algorithm.h gklx_algorithm.cc graph_handler.cc graph_handler.h \\
main.cc greedy_algorithm.cc greedy_algorithm.h random_handler.cc \\
random_handler.h nice_clustering_algorithm.cc nice_clustering_algorithm.h \\
-I/path/to/base/dir/ -O2 -o dynamic_facility_location

## The command to run the algorithm on input_file

`./dynamic_facility_location <input_file >output_file`

## Parameter setting

The main parameters of the algorithm can be set in the main.cc file:

*   num_levels_slack: the parameter beta of the agreement algorithm.
*   epsilon: the parameter lambda of the agreement algorithm.

The main parameters for generating the test instances can be set in the main.cc
file:

*   random_order: if set to true the input points are shuffled randomly before
    constructing the update order.
*   sliding_window_size: the size of the sliding window that determines the
    updates.
*   percentage_of_facilities: the percentage of the points to consider as
    facilities.
*   target_num_nodes: the number of points from the input file to consider in
    the instance.

## Input description

The input file is expected to first contain a line with two number separated by
a space "n d", where the first number n describes the number of points in the
entire instance, and the second number describes the number d of dimensions for
each point.

The first line is expected to be followed by n lines, where each line contains
the d-dimensional embedding as d numbers separated by a space.

## Output description

The output describes, for each of the six algorithms that are considered, three
values following each update:

*   The cost of the solution produced by the algorithm, in terms of the facility
    location objective.
*   The cumulative recourse of the algorithm, up until the last update.
*   The cumulative running time of the algorithm, up until the last update.

For more information, see the main.cc file.
