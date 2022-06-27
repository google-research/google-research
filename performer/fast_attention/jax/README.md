# Jax Variant of FAVOR+.

There are two main attention variants in the Jax variant:

* `make_fast_softmax_attention` - An unbiased and tight approximation of regular softmax attention. Can be used in Transformer models, as well as standalone for applications involving raw softmax attention or purely just softmax.
* `make_fast_generalized_attention` - Allows for generalized attention functions to produce different attention kernels as described in the paper.

The two functions create a `attention_fn` that has the same API as `flax.deprecated.nn.attention.dot_product_attention`, allowing quick replacement for a Transformer built on top of `flax.deprecated.nn.attention` modules.

The protein language modelling code can be found in [/google-research/protein_lm/](https://github.com/google-research/google-research/tree/master/protein_lm). In order to replace regular attention with our fast attention, set via gin: `FlaxModel.attention_fn = @make_fast_softmax_attention()` or `FlaxModel.attention_fn = @make_fast_generalized_attention()`.

## Notes:

* Set `lax_scan_unroll=16` for both attention functions when using a GPU to provide 4x speedups due to loop unrolling optimizations in the unidirectional case. However, set `lax_scan_unroll=1` (defaulted) when using a TPU.
* The unidirectional variant uses custom gradients via Jax, in order to provide significant memory reductions.
* This Jax version of FAVOR has also been integrated into the [Trax/Reformer library](https://github.com/google/trax/blob/master/trax/layers/research/sparsity.py) as `CausalFavor`, in order to provide additional memory gains via reversible layers.

