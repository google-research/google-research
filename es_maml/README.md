# ES-MAML Variant of Blackbox Optimization.

See "ES-MAML: Simple Hessian-Free Meta Learning" (https://arxiv.org/abs/1910.01215) for the paper associated with this library.

In order to run the algorithm, you must launch both the binaries es_maml_client (which produces the 'central' CPU) and multiple launches of es_maml_server (which produces the 'workers'). 
This depends on your particular distributed communication infrastructure, but we by default use GRPC.

The hyperparameters are all contained in config.py.

There are two algorithms:

1.  Zero Order
2.  First Order

## Zero Order:

1. Uses custom adaptation operators, built using blackbox algorithms such as MCBlackboxOptimizer, DPP sampling, and Hill-Climbing.

2. Collects state normalization data from all workers.

## First Order:

1.  Uses local-worker state normalization.

2.  Allows Hessian computation.
