# Performer's Fast Self Attention Module.

See ["Rethinking Attention with Performers"](https://arxiv.org/abs/2009.14794) for the paper associated with this library.

There are two main attention variants: 

* `make_fast_softmax_attention` - An unbiased and tight approximation of regular softmax attention. Can be used in Transformer models, as well as standalone for applications involving raw attention.
* `make_fast_generalized_attention` - Allows for generalized attention functions to produce different attention kernels as described in the paper.

The two functions create a `attention_fn` that has the same API as `flax.nn.attention.dot_product_attention`, allowing quick replacement for a Transformer built on top of `flax.nn.attention` modules.

Their default hyperparameters are currently optimal for a variety of tasks, such as protein modelling, image generation, and natural language processing.

The protein language modelling code can be found in [/google-research/protein_lm/](https://github.com/google-research/google-research/tree/master/protein_lm). In order to replace regular attention with our fast attention, set via gin: `FlaxModel.attention_fn = @make_fast_softmax_attention()` or `FlaxModel.attention_fn = @make_fast_generalized_attention()`.


## Notes:

* Set `lax_scan_unroll=16` for both attention functions when using a GPU to provide 4x speedups due to loop unrolling optimizations on the unidirectional case.
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
               Afroz Mohiuddin and
               Lukasz Kaiser and
               David Belanger and
               Lucy Colwell and
               Adrian Weller},
  title     = {Rethinking Attention with Performers},
  journal   = {CoRR},
  volume    = {abs/2009.14794},
  year      = {2020},
  url       = {https://arxiv.org/abs/2009.14794},
  archivePrefix = {arXiv},
  eprint    = {2009.14794}
}
```

