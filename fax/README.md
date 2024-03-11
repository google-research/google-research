# FAX

FAX is a software library designed to embed a federated programming model into
JAX. It is designed to make it easy to author and execute federated computations
in the datacenter, with a specific eye towards support large-scale federated
computations, including computations involving larger models, and ensuring that
they can be run efficiently. FAX embeds primitives like those defined by
TensorFlow Federated using the mapping capabilities and primitive extensions of
JAX.

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
