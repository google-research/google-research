# ES-ENAS: Evolutionary Strategies (ES) optimization combined with Efficient Neural Architecture Search (ENAS) for Reinforcement Learning Policies.

In order to run the algorithm, the user must launch both the binaries `client.py` (which produces the central 'aggregator') and multiple launches of `server.py` (which produces the 'workers'). The user must also implement their own client-server communication (see `client.py` and `server.py` for more details). GRPC may be used, for instance.

In addition to the normal ES setup, the client contains a PyGlove controller, which will suggest "topologies" (currently edges or partitions) to all workers, and collect back the reward each worker received in a weight sharing optimization process.

There are essentially two modes, with subclasses:

1.  Edge mode (`"NumpyEdgeSparsityPolicy"`). This generates a search space of 
    edges to prune for the NAS controller (client). Multiple search spaces are
    implemented in `make_search_space` function of the policy. The default is
    `aggregate_edges` which creates a feedforward network and only selects a
    fixed number (k) of edges. `independent_edges` chooses at each layer, a
    fixed number of edges to use.`residual_edges` allows residual layers.

2.  Partition mode (`"NumpyWeightSharingPolicy"`). This assigns each weight in 
    a normal network policy a "partition_index", and the number of partitions 
    is actually the true parameter vector.

`num_exact_evals` computes the current parameter/configurations's objective
value. Since the objective is now also and expectation over distribution of
topologies, it is defaulted to a much larger number than usual, e.g. 150.
