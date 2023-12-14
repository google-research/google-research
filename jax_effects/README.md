# jax_effects

Research project for algebraic effect handlers and choice-based learning in JAX.

This is the repository for the paper ["Choice-Based Learning in JAX"](https://openreview.net/pdf/0b8e24975d23827b5ad06eb61cdfcec70a486858.pdf).  Please cite this work as:

```
@inproceedings{
  2023jaxeffects,
  title={Choice-Based Learning in {JAX}},
  author={Shangyin Tan and Dan Zheng and Gordon Plotkin and Ningning Xie},
  booktitle={Machine Learning for Systems 2023},
  year={2023},
  url={https://openreview.net/forum?id=wkAFNdzhli}
}
```

*This is not an officially supported Google product.*

# Installation

```bash
# Download the repository.
git clone https://github.com/google-research/google-research
cd google-research/jax_effects

# Recommended: create a virtual environment.
virtualenv venv
source venv/bin/activate

# Install dependencies.
pip3 install -e .
```

# Examples

See `examples/` for example programs using `jax-effects`.

```bash
python3 -m jax_effects.examples.linear_regression
python3 -m jax_effects.examples.interleaved_effects
python3 -m jax_effects.examples.qlearning
```
