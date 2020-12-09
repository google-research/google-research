# Tensorflow Variant of FAVOR+.

The main attention function in the Tensorflow variant is `favor_attention`.

* In order to use softmax attention, set its argument `kernel_transformation=softmax_kernel_transformation`.
* In order to use generalized ReLU attention, set its argument `kernel_transformation=relu_kernel_transformation`. 

To use as a `tf.keras.layers.Layer` module, use the FAVOR `Attention` class instead (after setting the FAVOR+ configuration). This has a similar API to `tf.keras.layers.Attention()`.

## Notes:

* Just like the Jax case, the causal (unidirectional) variant uses custom gradients via `@tf.custom_gradient`, in order to provide significant memory reductions.


