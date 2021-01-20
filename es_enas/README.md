# ES-ENAS: Evolutionary Strategies (ES) optimization combined with Efficient Neural Architecture Search (ENAS) for Reinforcement Learning Policies.

See [ES-ENAS: Combining Evolution Strategies with Neural Architecture Search at No Extra Cost for Reinforcement Learning](https://arxiv.org/abs/2101.07415) for the associated paper.

In order to run the algorithm, the user must launch both the binaries `client.py` (which produces the central 'aggregator') and multiple launches of `server.py` (which produces the 'workers'). The user must also implement their own client-server communication (see `client.py` and `server.py` for more details). gRPC may be used, for instance; see the [open-sourced version of ES-MAML](https://github.com/google-research/google-research/tree/master/es_maml) on how to use gRPC.

In addition to the normal ES setup, the client contains a [PyGlove](https://proceedings.neurips.cc/paper/2020/hash/012a91467f210472fab4e11359bbfef6-Abstract.html) controller, which will suggest "topologies" (currently edges or partitions) to all workers, and collect back the reward each worker received in a weight sharing optimization process. Currently, PyGlove is not open-sourced yet, but will be in the future. The code can be treated as a template for performing the ES-ENAS algorithm.

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

If you found this codebase useful, please consider citing the paper:

```
@inproceedings{es_enas,
  author    = {Xingyou Song and
               Krzysztof Choromanski and
               Jack Parker-Holder and
               Yunhao Tang and
               Daiyi Peng and
               Deepali Jain and
               Wenbo Gao and
               Aldo Pacchiano and
               Tamas Sarlos and
               Yuxiang Yang}, 
  title     = {ES-ENAS: Combining Evolution Strategies with Neural Architecture Search at No Extra Cost for Reinforcement Learning},
  journal   = {CoRR},
  volume    = {abs/2101.07415},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.07415},
  archivePrefix = {arXiv},
  eprint    = {2101.07415},
}
```
Thanks!
