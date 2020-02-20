# Algorithms for truss decomposition

`truss_decomposition_parallel.cpp` contains code to compute the truss
decomposition of a graph.

`td_approx_external.cpp` contains an external-memory implementation of an
algorithm to approximate the size of the max-truss of a graph.

The algorithms implemented here are described in
A. Conte, A. Marino, D. De Sensi, R. Grossi, L. Versari
"Truly Scalable k-truss and max-truss Algorithms for Community Detection in
Graphs"

## Compiling and running
Execute `./compile.sh` to compile the code. `gflags` needs to be installed on
your system. On debian-based distributions, this can be achieved by running

  apt install libgflags-dev

Input is accepted by default from standard input in the following format:

- first line contains `N`, the number of nodes.
- the next `N` lines contain two numbers, `i` and `degree[i]`
- all the other lines contain two numbers, `a[i]` and `b[i]`, representing an
  edge from node `a[i]` to node `b[i]`.

For more parameters, run `./td_approx_external --help` or
`./truss_decomposition_parallel --help`.

Run `./run.sh` for an example execution on the provided `clique10.nde` file.
