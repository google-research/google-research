# FAX - Scalable and Differentiable Federated Primitives in JAX

FAX is a software library designed to embed a federated programming model into
JAX. FAX has multiple objectives.

1. Create a JAX-based authoring surface for distributed and federated
computations.
1. Leverage JAX's sharding mechanisms to enable highly optimized execution,
especially in large-scale datacenter settings.
1. Provide an efficient implementation of
[federated automatic differentiation](https://arxiv.org/abs/2301.07806), to make
it easier to design and express new federated algorithms.

FAX is designed to make it easy to author and execute federated computations
in the datacenter. FAX is tailored towards **large-scale** federated
computations, including computations involving larger models, and ensuring that
they can be run efficiently. FAX embeds primitives like those defined by
[TensorFlow Federated](https://github.com/tensorflow/federated) using the
mapping capabilities and primitive extensions of JAX.

For an introduction on how to use FAX, check out the colab notebook in the
`tutorials` subdirectory.

## Installing

```
pip install --upgrade google-fax
```

## Building a new wheel

Run `python -m build` to build a new `google-fax` wheel.

## Run tests

Execute the `run_test.sh` script.
