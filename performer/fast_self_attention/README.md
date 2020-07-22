# Performer's Fast Self Attention Module.

See ["Masked Language Modeling for Proteins via Linearly Scalable Long-Context Transformers"](https://arxiv.org/abs/2006.03555) for the paper associated with this library.

There are two main attention variants, a `make_fast_softmax_attention` and a `make_fast_generalized_attention`. `make_fast_softmax_attention` is an unbiased approximation of regular softmax attention, while `make_fast_generalized_attention` allows for generalized attention functions as described in the paper. Their default hyperparameters are currently optimal for the task of protein language modelling. The two functions create a `attention_fn` that has the same API as `flax.nn.attention.dot_product_attention`, allowing quick replacement for a Transformer built on top of `flax.nn.attention` modules.

If you found this codebase useful, please consider citing the paper:

```
@article{performer,
  author    = {Krzysztof Choromanski and
               Valerii Likhosherstov and
               David Dohan and
               Xingyou Song and
               Jared Davis and
               Tam{\'{a}}s Sarl{\'{o}}s and
               David Belanger and
               Lucy Colwell and
               Adrian Weller},
  title     = {Masked Language Modeling for Proteins via Linearly Scalable Long-Context
               Transformers},
  journal   = {CoRR},
  volume    = {abs/2006.03555},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.03555},
  archivePrefix = {arXiv},
  eprint    = {2006.03555}
}
```
