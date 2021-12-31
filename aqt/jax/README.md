# Quantization library for Accurate Quantized Training

`Jax` and `Flax` quantization libraries provides `what you serve is what you train`
quantization for convolution and matmul.

`jax/imagenet` directory contains quantized ResNet model.

### Jax Libraries

- **quantization.quantized_dot**: [LAX.dot](https://github.com/google/jax/blob/f65a327c764406db45e95048dfe09209d8ef6d37/jax/_src/lax/lax.py#L632) with optionally quantized weights and activations.
- **quantization.quantized_dynamic_dot_general**: [LAX.dot general](https://github.com/google/jax/blob/f65a327c764406db45e95048dfe09209d8ef6d37/jax/_src/lax/lax.py#L667) with optionally quantized dynamic inputs.
- **quantization.quantized_sum**: Sums a tensor while quantizing intermediate accumulations.
- **quantization.dot_general_aqt**: Adds quantization to [LAX.dot_general](https://github.com/google/jax/blob/f65a327c764406db45e95048dfe09209d8ef6d37/jax/_src/lax/lax.py#L667) with option to use integer dot.


### Flax Libraries

- **flax_layers.DenseAqt**: Adds quantization to [Flax Dot Module](https://github.com/google/flax/blob/65061e6128f6695eed441acf2bfffc3b1badd318/flax/nn/linear.py#L134)
- **flax_layers.ConvAqt**: Adds quantization to [Flax Conv Module](https://github.com/google/flax/blob/65061e6128f6695eed441acf2bfffc3b1badd318/flax/nn/linear.py#L189)
- **flax_layers.EmbedAqt**: Adds quantization to [Flax Embed Module](https://github.com/google/flax/blob/65061e6128f6695eed441acf2bfffc3b1badd318/flax/nn/linear.py#L360).
- **flax_layers.LayerNormAqt**: Adds quantization support to the [Flax LayerNorm layer](https://github.com/google/flax/blob/65061e6128f6695eed441acf2bfffc3b1badd318/flax/linen/normalization.py#L140)
- **flax_attention.MultiHeadDotProductAttentionAqt**: Adds quantization to [Flax Multi-head dot-product attention](https://github.com/google/flax/blob/65061e6128f6695eed441acf2bfffc3b1badd318/flax/nn/attention.py#L206).


