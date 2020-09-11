# Shared-memory Parallel Clustering

This repository contains shared-memory parallel clustering algorithms. It
currently consists of affinity clustering and correlation clustering. Note
that the repository uses the
[Graph-Based Benchmark Suite (GBBS)](https://github.com/ParAlg/gbbs)
for parallel primitives and benchmarks.

## Installation

Compiler: g++ &gt;= 7.4.0 with support for Cilk Plus, or g++ &gt;= 7.4.0 with
pthread support (to use the Homegrown scheduler)

Build system: [Bazel](https:://bazel.build)

Note that the parallel scheduler can be chosen using comand-line parameters.
The default compilation is serial, and the parameters HOMEGROWN, CILK, and
OPENMP switch between the Homegrown scheduler, Cilk Plus, and OpenMP
respectively.

## Example Usage

The main executable is cluster-in-memory_main. It will run either parallel
affinity clusterer or parallel correlation clusterer, by setting the input
clusterer_name to "ParallelAffinityClusterer" or "ParallelCorrelationClusterer"
respectively. Both clusterers take configs given in a protobuf, as detailed
in config.proto, which can be passed in by setting the input clusterer_config.
Note that the proto should be passed in text format.

The input graph format should be an edge list format. For an unweighted graph,
every line consists of two endpoints separated by a space, and the weight of
each edge with be set to 1. For a weighted graph, every line consists of
two endpoints, followed by a weight, each separated by a space. The relevant
inputs --symmetric_graph and --float_weighted can be toggled to denote whether
the graph is already symmetrized / needs to be symmetrized and is weighted /
unweighted respectively. The output clustering format gives one vertex per
line, and a double line break marking the boundaries between different clusters.

Note that various flags must be set to ensure compilation, due to dependencies
on GBBS. A template command is:

```
bazel run -c opt --cxxopt=-pthread --cxxopt=-std=c++17 --cxxopt=-w --cxxopt=-mcx16 --cxxopt=-march=native --cxxopt=-fvisibility=hidden --cxxopt=-fvisibility-inlines-hidden --cxxopt=-DHOMEGROWN --cxxopt=-DLONG --cxxopt=-DUSEMALLOC :cluster-in-memory_main -- --input_graph=</path/to/graph> --output_clustering=</path/to/output> --clusterer_name=<clusterer name> --clusterer_config='<clusterer proto>'
```
