# ES-MAML Variant of Blackbox Optimization.

See ["ES-MAML: Simple Hessian-Free Meta Learning"](https://arxiv.org/abs/1910.01215) for the paper associated with this library. This was also used in ["Rapidly Adaptable Legged Robots via Evolutionary Meta-Learning"](https://arxiv.org/abs/2003.01239) with associated [Google AI Blog Post](https://ai.googleblog.com/2020/04/exploring-evolutionary-meta-learning-in.html).

In order to run the algorithm, you must launch both the binaries `es_maml_client` (which produces the central 'aggregator') and multiple launches of `es_maml_server` (which produces the 'workers').

This depends on your particular distributed communication infrastructure, but we by default use GRPC. In order to use the default GRPC method of client-server communication, you must first create the proper `pb2.py` and `pb2_grpc.py` libraries from the `.proto`'s for both `zero_order` and `first_order`. This can be done via the commands (see [discussion](https://github.com/google-research/google-research/issues/499)):

```
$ pip install protobuf
$ pip install grpcio-tools==1.32
$ pip install googleapis-common-protos

$ python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. first_order.proto
$ python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. zero_order.proto
```

## Algorithms

The hyperparameters are all contained in `config.py`.

There are two algorithms:

1.  Zero Order
2.  First Order

### Zero Order:

1. Uses custom adaptation operators, built using blackbox algorithms such as MCBlackboxOptimizer, DPP sampling, and Hill-Climbing.

2. Collects state normalization data from all workers.

### First Order:

1.  Uses local-worker state normalization.

2.  Allows Hessian computation.

If you found this codebase useful, please consider citing the two papers:

```
@inproceedings{es_maml,
  author    = {Xingyou Song and
               Wenbo Gao and
               Yuxiang Yang and
               Krzysztof Choromanski and
               Aldo Pacchiano and
               Yunhao Tang},
  title     = {{ES-MAML:} Simple Hessian-Free Meta Learning},
  booktitle = {8th International Conference on Learning Representations, {ICLR} 2020,
               Addis Ababa, Ethiopia, April 26-30, 2020},
  year      = {2020},
  url       = {https://openreview.net/forum?id=S1exA2NtDB},
}

@article{rapidly,
  author    = {Xingyou Song and
               Yuxiang Yang and
               Krzysztof Choromanski and
               Ken Caluwaerts and
               Wenbo Gao and
               Chelsea Finn and
               Jie Tan},
  title     = {Rapidly Adaptable Legged Robots via Evolutionary Meta-Learning},
  booktitle = {International Conference on Intelligent Robots and Systems, {IROS} 2020},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.01239},
}

```
Thanks!
