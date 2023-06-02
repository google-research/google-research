# Arithmetic Sampling

This codebase allows for use of the
[Arithmetic Sampling](https://arxiv.org/abs/2210.15458) algorithm for sampling 
from sequence models in T5X.

## Introduction

[Arithmetic Sampling](https://arxiv.org/abs/2210.15458) is an algorithm for
sampling from sequence models that provides provably increased beam diversity
compared to regular sampling in some situations, as well as lowered estimator
variance. The algorithm is also parallelizable.

## How to use arithmetic sampling

This library provides a T5X implementation of the algorithm for use with any
model that can accept an `EncoderDecoderModel.decode_fn`, though implementations
for other model types should be quite straightforward. The gin files in this
library can be included in any compatible T5X model to use arithmetic sampling.

The easiest way to get started on accelerators is to plug one of the included 
gin configs into the 
[T5X Quickstart guide](https://github.com/google-research/t5x).

Parallel decoding can be accomplished by pre-computing the codes for each
sample, fixing the RNG seed, and passing them in batches along with the codes.

The included `run.sh` will install locally (including installing t5x from 
GitHub) and run the tests with a fallback to CPU mode.
