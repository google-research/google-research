# Performer's Fast Self Attention Module.

See ["Masked Language Modeling for Proteins via Linearly Scalable Long-Context Transformers"](https://arxiv.org/abs/2006.03555) for the paper associated with this library.

There are two main attention variants, a `make_fast_softmax_attention` and a `make_fast_generalized_attention`. `make_fast_softmax_attention` is an unbiased approximation of regular softmax attention, while `make_fast_generalized_attention` allows for generalized attention functions as described in the paper. Their default hyperparameters are currently optimal for the task of protein language modelling. The two functions create a `attention_fn` that has the same API as `flax.nn.attention.dot_product_attention`, allowing quick replacement for a Transformer built on top of `flax.nn.attention` modules.

The protein language modelling code can be found in [/google-research/protein_lm/](https://github.com/google-research/google-research/tree/master/protein_lm). In order to replace regular attention with our fast attention, set via gin: `FlaxModel.attention_fn=@make_fast_softmax_attention()` or `FlaxModel.attention_fn=@make_fast_generalized_attention()`.


## Notes:

* Set `lax_scan_unroll=16` for both attention functions when using a GPU to provide 4x speedups due to loop unrolling optimizations.
* The unidirectional variant uses custom gradients via Jax, in order to provide significant memory reductions.

If you found this codebase useful, please consider citing the paper:

```
@article{performer,
  author    = {Krzysztof Choromanski and
               Valerii Likhosherstov and
               David Dohan and
               Xingyou Song and
               Andreea Gane and
               Tam{\'{a}}s Sarl{\'{o}}s and
               Peter Hawkins and
               Jared Davis and
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

