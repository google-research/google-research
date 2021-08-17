# Zebraix

## Important note

Zebraix is an entirely personal project. It is not endorsed, or in any other way
given approval, by any organization or person.

## Running tests and updating goldens

To run tests do something like:
```
bazel test -c dbg ${ZERBRAIX_DIR}/testdata/samples:all \
   --test_arg=--diff_lines=10000 --nocache_test_results
```

To update goldens do something like:
```
bazel test --use_sponge=no -c dbg ${ZERBRAIX_DIR}/testdata/samples:all \
   --test_arg="--update_dir=$(PWD_CMD_HERE)" --test_strategy=local
```

## Overview

Zebraix is a set of utilities for handling graphs that can be described as
partially ordered sets with order-dimension 2. The general case is described
well in the Wikipedia article about
[dominance drawings](https://en.wikipedia.org/wiki/Dominance_drawing).

The real interest of Zebraix is a special case of DAGs, of which trees are a
further special case. These are

*   DAGs, that are
*   planar, that are
*   basically like st-planar graphs, and that are
*   transitively irreducible, but
*   need not actually have a source and a sink, yet nonetheless
*   a source can be extrapolated by connecting it to all nodes without incoming
    edges, and
*   a sink can be extrapolated by connecting it to all nodes without outgoing
    edges.

In order for the extrapolation to work, the graph specification must give the
order of children and (by implication) parents.

We refer to the graphs that Zebraix handles as *Jaywalks*. The above description
is of a planar Jaywalk. This is what we normally mean. The more general case is
a *potentially non-planar Jaywalk* or, when it is known that there are edge
crossings, a *non-planar Jaywalk*.

## Scope

The following are core capabilities for Zebraix.

*   A protobuf or protobuf-like text format for defining Jaywalks, with reading
    and writing methods. The text format is convenient for specifying a graph as
    a partially ordered set with order-dimension 2.
*   Algorithms for building graph structures from the Jaywalk text
    specification.
*   Additional provision for building co-dominance graphs.
*   Capabilities for rendering graphs as SVG pictures.
*   Strong capabilities for handling trees, both in rendering and efficiency of
    processing.
*   Graceful increase in complexity with more complex graphs, both in rendering
    and efficiency of processing.
