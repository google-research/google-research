# JaxSel: a differentiable subgraph extraction layer, in Jax

The `jaxsel` module contains a sparse subgraph selection layer, implementing a
differentiable sparse PageRank. The goal of our model is to extract a small
subgraph from a large graph, where the subgraph is enough to solve a given task.
As such, we consider datasets of graphs. We provide an API for modeling
directed, unweighted graphs, which allows querying neighbors of a given node in
the graph, and querying node features and edge features. All functionalities are
local.

Given a graph, our pipeline contains the following three (3) blocks:

*   An Agent model, which can estimate weights for the graph edges, using local
    features. The Agent model can be used to implicitly represent a full
    weighted adjacency matrix over the underlying unweighted, directed graph.
*   A differentiable subgraph selection layer. This is our main contribution:
    our layer learns to select a subgraph around a given start node, using an
    underlying Agent model.
*   A downstream graph neural network. This model takes the extracted subgraph,
    and makes a prediction for the desired task. Currently, we only handle graph
    classification.

## Set up

Launch a test run by running `./run.sh` from the `jaxsel` folder. The script
will clone the
[long-range-arena](https://github.com/google-research/long-range-arena)
repository, which we require.

If this does not work, make sure you are in the `jaxsel` folder.

To train a model on the pathfinder dataset, you must first download the data,
following instructions
[here](https://github.com/google-research/long-range-arena). Then, you must
specify the path to the dataset in the `_PATHFINDER_TFDS_PATH` variable in
`jaxsel/examples/pathfinder_data.py`.

## Launching an experiment

We currently have an example in `examples/train.py`. This example handles
image-based classification tasks. For now, we have implemented data loading for
MNIST and for the Pathfinder datasets from the Long Range Attention benchmark
suite.

Launch an experiment by running: `python -m jaxsel.examples.train --dataset
mnist` from the `google_research` directory.
