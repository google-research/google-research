# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2019 The TensorFlow Probability Authors.
# Copyright 2019 OpenAI (http://openai.com).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The Pixel CNN++ distribution class.

The script is copied from and modified based on
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/pixel_cnn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import mixture_same_family
from tensorflow_probability.python.distributions import quantized_distribution
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.layers import weight_norm

tf.compat.v1.disable_v2_behavior()


class PixelCNN(distribution.Distribution):
  """The Pixel CNN++ distribution.

  Pixel CNN++ [(Salimans et al., 2017)][1] models a distribution over image
  data, parameterized by a neural network. It builds on Pixel CNN and
  Conditional Pixel CNN, as originally proposed by [(van den Oord et al.,
  2016)][2, 3]. The model expresses the joint distribution over pixels as
  the product of conditional distributions: `p(x|h) = prod{ p(x[i] | x[0:i], h)
  : i=0, ..., d }`, in which `p(x[i] | x[0:i], h) : i=0, ..., d` is the
  probability of the `i`-th pixel conditional on the pixels that preceded it in
  raster order (color channels in RGB order, then left to right, then top to
  bottom). `h` is optional additional data on which to condition the image
  distribution, such as class labels or VAE embeddings. The Pixel CNN++
  network enforces the dependency structure among pixels by applying a mask to
  the kernels of the convolutional layers that ensures that the values for each
  pixel depend only on other pixels up and to the left (see
  `tfd.PixelCnnNetwork`).

  Pixel values are modeled with a mixture of quantized logistic distributions,
  which can take on a set of distinct integer values (e.g. between 0 and 255
  for an 8-bit image).

  Color intensity `v` of each pixel is modeled as:

  `v ~ sum{q[i] * quantized_logistic(loc[i], scale[i]) : i = 0, ..., k },

  in which `k` is the number of mixture components and the `q[i]` are the
  Categorical probabilities over the components.

  #### Sampling

  Pixels are sampled one at a time, in raster order. This enforces the
  autoregressive dependency structure, in which the sample of pixel `i` is
  conditioned on the samples of pixels `1, ..., i-1`. A single color image is
  sampled as follows:

  ```python
  samples = random_uniform([image_height, image_width, image_channels])
  for i in image_height:
    for j in image_width:
      component_logits, locs, scales, coeffs = pixel_cnn_network(samples)
      components = Categorical(component_logits).sample()
      locs = gather(locs, components)
      scales = gather(scales, components)

      coef_count = 0
      channel_samples = []
      for k in image_channels:
        loc = locs[k]
        for m in range(k):
          loc += channel_samples[m] * coeffs[coef_count]
          coef_count += 1
        channel_samp = Logistic(loc, scales[k]).sample()
        channel_samples.append(channel_samp)
      samples[i, j, :] = tf.stack(channel_samples, axis=-1)
  samples = round(samples)
  ```

  #### Examples

  ```python

  # Build a small Pixel CNN++ model to train on MNIST.

  import tensorflow as tf
  import tensorflow_datasets as tfds
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  tfk = tf.keras
  tfkl = tf.keras.layers

  tf.enable_v2_behavior()

  # Load MNIST from tensorflow_datasets
  data = tfds.load('mnist')
  train_data, test_data = data['train'], data['test']

  def image_preprocess(x):
    x['image'] = tf.cast(x['image'], tf.float32)
    return (x['image'],)  # (input, output) of the model

  batch_size = 16
  train_it = train_data.map(image_preprocess).batch(batch_size).shuffle(1000)

  # Define a Pixel CNN network
  dist = tfd.PixelCNN(
      image_shape=(28, 28, 1),
      num_resnet=1,
      num_hierarchies=2,
      num_filters=32,
      num_logistic_mix=5,
      dropout_p=.3,
  )

  # Define the model input
  image_input = tfkl.Input(shape=input_shape)

  # Define the log likelihood for the loss fn
  log_prob = dist.log_prob(image_input)

  # Define the model
  model = tfk.Model(inputs=image_input, outputs=log_prob)
  model.add_loss(-tf.reduce_mean(log_prob))

  # Compile and train the model
  model.compile(
      optimizer=tfk.optimizers.Adam(.001),
      metrics=[])

  model.fit(train_it, epochs=10, verbose=True)

  # sample five images from the trained model
  samples = dist.sample(5)

  ```

  To train a class-conditional model:

  ```python

  data = tfds.load('mnist')
  train_data, test_data = data['train'], data['test']

  def image_preprocess(x):
    x['image'] = tf.cast(x['image'], tf.float32)
    # return model (inputs, outputs): inputs are (image, label) and there are no
    # outputs
    return ((x['image'], x['label']),)

  batch_size = 16
  train_ds = train_data.map(image_preprocess).batch(batch_size).shuffle(1000)
  optimizer = tfk.optimizers.Adam()

  image_shape = (28, 28, 1)
  label_shape = ()
  dist = tfd.PixelCNN(
      image_shape=image_shape,
      conditional_shape=label_shape,
      num_resnet=1,
      num_hierarchies=2,
      num_filters=32,
      num_logistic_mix=5,
      dropout_p=.3,
  )

  image_input = tfkl.Input(shape=image_shape)
  label_input = tfkl.Input(shape=label_shape)

  log_prob = dist.log_prob(image_input, conditional_input=label_input)

  class_cond_model = tfk.Model(
      inputs=[image_input, label_input], outputs=log_prob)
  class_cond_model.add_loss(-tf.reduce_mean(log_prob))
  class_cond_model.compile(
      optimizer=tfk.optimizers.Adam(),
      metrics=[])
  class_cond_model.fit(train_ds, epochs=10)

  # Take 10 samples of the digit '5'
  samples = dist.sample(10, conditional_input=5.)

  # Take 4 samples each of the digits '1', '2', '3'.
  # Note that when a batch of conditional input is passed, the sample shape
  # (the first argument of `dist.sample`) must have its last dimension(s) equal
  # the batch shape of the conditional input (here, (3,)).
  samples = dist.sample((4, 3), conditional_input=[1., 2., 3.])

  ```

  Note: PixelCNN may also be trained using tfp.layers.DistributionLambda;
  however, as of this writing, that method is much slower and has the
  disadvantage of calling `sample()` upon construction, which causes the
  `PixelCnnNetwork` to be initialized with random data (if data-dependent
  initialization is used).

  #### References

  [1]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.
       PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture
       Likelihood and Other Modifications. In _International Conference on
       Learning Representations_, 2017.
       https://pdfs.semanticscholar.org/9e90/6792f67cbdda7b7777b69284a81044857656.pdf
       Additional details at https://github.com/openai/pixel-cnn

  [2]: Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt,
       Alex Graves, and Koray Kavukcuoglu. Conditional Image Generation with
       PixelCNN Decoders. In _Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1606.05328

  [3]: Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel
       Recurrent Neural Networks. In _International Conference on Machine
       Learning_, 2016. https://arxiv.org/pdf/1601.06759.pdf
  """

  def __init__(self,
               image_shape,
               conditional_shape=None,
               num_resnet=5,
               num_hierarchies=3,
               num_filters=160,
               num_logistic_mix=10,
               receptive_field_dims=(3, 3),
               dropout_p=0.5,
               resnet_activation='concat_elu',
               reg_weight=0.0,
               use_weight_norm=True,
               use_data_init=True,
               high=255,
               low=0,
               rescale_pixel_value=True,
               dtype=tf.float32,
               name='PixelCNN'):
    """Construct Pixel CNN++ distribution.

    Args:
      image_shape: 3D `TensorShape` or tuple for the `[height, width, channels]`
        dimensions of the image.
      conditional_shape: `TensorShape` or tuple for the shape of the conditional
        input, or `None` if there is no conditional input.
      num_resnet: `int`, the number of layers (shown in Figure 2 of [2]) within
        each highest-level block of Figure 2 of [1].
      num_hierarchies: `int`, the number of hightest-level blocks (separated by
        expansions/contractions of dimensions in Figure 2 of [1].)
      num_filters: `int`, the number of convolutional filters.
      num_logistic_mix: `int`, number of components in the logistic mixture
        distribution.
      receptive_field_dims: `tuple`, height and width in pixels of the receptive
        field of the convolutional layers above and to the left of a given
        pixel. The width (second element of the tuple) should be odd. Figure 1
        (middle) of [2] shows a receptive field of (3, 5) (the row containing
        the current pixel is included in the height). The default of (3, 3) was
        used to produce the results in [1].
      dropout_p: `float`, the dropout probability. Should be between 0 and 1.
      resnet_activation: `string`, the type of activation to use in the resnet
        blocks. May be 'concat_elu', 'elu', or 'relu'.
      reg_weight: 'float', the l2 regularization.
      use_weight_norm: `bool`, if `True` then use weight normalization (works
        only in Eager mode).
      use_data_init: `bool`, if `True` then use data-dependent initialization
        (has no effect if `use_weight_norm` is `False`).
      high: `int`, the maximum value of the input data (255 for an 8-bit image).
      low: `int`, the minimum value of the input data.
      rescale_pixel_value: 'bool', if `True` then rescale pixel value to [-1,1].
      dtype: Data type of the `Distribution`.
      name: `string`, the name of the `Distribution`.
    """

    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(PixelCNN, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=False,
          allow_nan_stats=True,
          parameters=parameters,
          name=name)

      if not tensorshape_util.is_fully_defined(image_shape):
        raise ValueError('`image_shape` must be fully defined.')

      if (conditional_shape is not None and
          not tensorshape_util.is_fully_defined(conditional_shape)):
        raise ValueError('`conditional_shape` must be fully defined`')

      if tensorshape_util.rank(image_shape) != 3:
        raise ValueError('`image_shape` must have length 3, representing '
                         '[height, width, channels] dimensions.')

      self._high = tf.cast(high, self.dtype)
      self._low = tf.cast(low, self.dtype)
      self._rescale_pixel_value = rescale_pixel_value
      self._num_logistic_mix = num_logistic_mix
      self.network = _PixelCNNNetwork(
          dropout_p=dropout_p,
          num_resnet=num_resnet,
          num_hierarchies=num_hierarchies,
          num_filters=num_filters,
          num_logistic_mix=num_logistic_mix,
          receptive_field_dims=receptive_field_dims,
          resnet_activation=resnet_activation,
          reg_weight=reg_weight,
          use_weight_norm=use_weight_norm,
          use_data_init=use_data_init,
          high=high,
          low=low,
          rescale_pixel_value=rescale_pixel_value,
          dtype=dtype)

      image_input_shape = tensorshape_util.concatenate([None], image_shape)
      if conditional_shape is None:
        input_shape = image_input_shape
      else:
        conditional_input_shape = tensorshape_util.concatenate(
            [None], conditional_shape)
        input_shape = [image_input_shape, conditional_input_shape]

      self.image_shape = image_shape
      self.conditional_shape = conditional_shape
      self.network.build(input_shape)

  def _make_mixture_dist(self,
                         component_logits,
                         locs,
                         scales,
                         return_per_pixel=False):
    """Builds a mixture of quantized logistic distributions.

    Args:
      component_logits: 4D `Tensor` of logits for the Categorical distribution
        over Quantized Logistic mixture components. Dimensions are `[batch_size,
        height, width, num_logistic_mix]`.
      locs: 4D `Tensor` of location parameters for the Quantized Logistic
        mixture components. Dimensions are `[batch_size, height, width,
        num_logistic_mix, num_channels]`.
      scales: 4D `Tensor` of location parameters for the Quantized Logistic
        mixture components. Dimensions are `[batch_size, height, width,
        num_logistic_mix, num_channels]`.
      return_per_pixel: `bool`. If True, return per pixel level log prob.

    Returns:
      dist: A quantized logistic mixture `tfp.distribution` over the input data.
    """
    mixture_distribution = categorical.Categorical(logits=component_logits)

    if self._rescale_pixel_value:
      # Convert distribution parameters for pixel values in
      # `[self._low, self._high]` for use with `QuantizedDistribution`
      locs = self._low + 0.5 * (self._high - self._low) * (locs + 1.)
      scales *= 0.5 * (self._high - self._low)

    logistic_dist = quantized_distribution.QuantizedDistribution(
        distribution=transformed_distribution.TransformedDistribution(
            distribution=logistic.Logistic(loc=locs, scale=scales),
            bijector=shift.Shift(shift=tf.cast(-0.5, self.dtype))),
        low=self._low,
        high=self._high)

    dist = mixture_same_family.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=independent.Independent(
            logistic_dist, reinterpreted_batch_ndims=1))
    if return_per_pixel:
      return dist
    else:
      return independent.Independent(dist, reinterpreted_batch_ndims=2)

  def _log_prob(self,
                value,
                conditional_input=None,
                training=None,
                return_per_pixel=False):
    """Log probability function with optional conditional input.

    Calculates the log probability of a batch of data under the modeled
    distribution (or conditional distribution, if conditional input is
    provided).

    Args:
      value: `Tensor` or Numpy array of image data. May have leading batch
        dimension(s), which must broadcast to the leading batch dimensions of
        `conditional_input`.
      conditional_input: `Tensor` on which to condition the distribution (e.g.
        class labels), or `None`. May have leading batch dimension(s), which
        must broadcast to the leading batch dimensions of `value`.
      training: `bool` or `None`. If `bool`, it controls the dropout layer,
        where `True` implies dropout is active. If `None`, it defaults to
        `tf.keras.backend.learning_phase()`.
      return_per_pixel: `bool`. If True, return per pixel level log prob.

    Returns:
      log_prob_values: `Tensor`.
    """
    # Determine the batch shape of the input images
    image_batch_shape = prefer_static.shape(value)[:-3]

    # Broadcast `value` and `conditional_input` to the same batch_shape
    if conditional_input is None:
      image_batch_and_conditional_shape = image_batch_shape
    else:
      conditional_input = tf.convert_to_tensor(conditional_input)
      conditional_input_shape = prefer_static.shape(conditional_input)
      conditional_batch_rank = (prefer_static.rank(conditional_input) -
                                tensorshape_util.rank(self.conditional_shape))
      conditional_batch_shape = conditional_input_shape[:conditional_batch_rank]

      image_batch_and_conditional_shape = prefer_static.broadcast_shape(
          image_batch_shape, conditional_batch_shape)
      conditional_input = tf.broadcast_to(
          conditional_input,
          prefer_static.concat(
              [image_batch_and_conditional_shape, self.conditional_shape],
              axis=0))
      value = tf.broadcast_to(
          value,
          prefer_static.concat(
              [image_batch_and_conditional_shape, self.event_shape],
              axis=0))

      # Flatten batch dimension for input to Keras model
      conditional_input = tf.reshape(
          conditional_input,
          prefer_static.concat([(-1,), self.conditional_shape], axis=0))

    value = tf.reshape(
        value, prefer_static.concat([(-1,), self.event_shape], axis=0))

    if self._rescale_pixel_value:
      transformed_value = (2. * (value - self._low) /
                           (self._high - self._low)) - 1.
      inputs = (
          transformed_value if conditional_input is None else
          [transformed_value, conditional_input])
    else:
      inputs = (
          value if conditional_input is None else [value, conditional_input])

    params = self.network(inputs, training=training)

    num_channels = self.event_shape[-1]
    if num_channels == 1:
      component_logits, locs, scales = params
    else:
      # If there is more than one channel, we create a linear autoregressive
      # dependency among the location parameters of the channels of a single
      # pixel (the scale parameters within a pixel are independent). For a pixel
      # with R/G/B channels, the `r`, `g`, and `b` saturation values are
      # distributed as:
      #
      # r ~ Logistic(loc_r, scale_r)
      # g ~ Logistic(coef_rg * r + loc_g, scale_g)
      # b ~ Logistic(coef_rb * r + coef_gb * g + loc_b, scale_b)
      # TODO(emilyaf) Investigate using fill_triangular/matrix multiplication
      # on the coefficients instead of split/multiply/concat
      component_logits, locs, scales, coeffs = params
      num_coeffs = num_channels * (num_channels - 1) // 2
      loc_tensors = tf.split(locs, num_channels, axis=-1)
      coef_tensors = tf.split(coeffs, num_coeffs, axis=-1)
      channel_tensors = tf.split(value, num_channels, axis=-1)

      coef_count = 0
      for i in range(num_channels):
        channel_tensors[i] = channel_tensors[i][Ellipsis, tf.newaxis, :]
        for j in range(i):
          loc_tensors[i] += channel_tensors[j] * coef_tensors[coef_count]
          coef_count += 1
      locs = tf.concat(loc_tensors, axis=-1)

    dist = self._make_mixture_dist(
        component_logits, locs, scales, return_per_pixel=return_per_pixel)
    log_px = dist.log_prob(value)
    if return_per_pixel:
      return log_px
    else:
      return tf.reshape(log_px, image_batch_and_conditional_shape)

  def _sample_n(self, n, seed=None, conditional_input=None, training=False):
    """Samples from the distribution, with optional conditional input.

    Args:
      n: `int`, number of samples desired.
      seed: `int`, seed for RNG. Setting a random seed enforces reproducability
        of the samples between sessions (not within a single session).
      conditional_input: `Tensor` on which to condition the distribution (e.g.
        class labels), or `None`.
      training: `bool` or `None`. If `bool`, it controls the dropout layer,
        where `True` implies dropout is active. If `None`, it defers to Keras'
        handling of train/eval status.
    Returns:
      samples: a `Tensor` of shape `[n, height, width, num_channels]`.
    """
    if conditional_input is not None:
      conditional_input = tf.convert_to_tensor(
          conditional_input, dtype=self.dtype)
      conditional_event_rank = tensorshape_util.rank(self.conditional_shape)
      conditional_input_shape = prefer_static.shape(conditional_input)
      conditional_sample_rank = prefer_static.rank(
          conditional_input) - conditional_event_rank

      # If `conditional_input` has no sample dimensions, prepend a sample
      # dimension
      if conditional_sample_rank == 0:
        conditional_input = conditional_input[tf.newaxis, Ellipsis]
        conditional_sample_rank = 1

      # Assert that the conditional event shape in the `PixelCnnNetwork` is the
      # same as that implied by `conditional_input`.
      conditional_event_shape = conditional_input_shape[
          conditional_sample_rank:]
      with tf.control_dependencies([
          tf.assert_equal(self.conditional_shape, conditional_event_shape)]):

        conditional_sample_shape = conditional_input_shape[
            :conditional_sample_rank]
        repeat = n // prefer_static.reduce_prod(conditional_sample_shape)
        h = tf.reshape(
            conditional_input,
            prefer_static.concat([(-1,), self.conditional_shape], axis=0))
        h = tf.tile(h,
                    prefer_static.pad(
                        [repeat], paddings=[[0, conditional_event_rank]],
                        constant_values=1))

    samples_0 = tf.random.uniform(
        prefer_static.concat([(n,), self.event_shape], axis=0),
        minval=-1., maxval=1., dtype=self.dtype, seed=seed)
    inputs = samples_0 if conditional_input is None else [samples_0, h]
    params_0 = self.network(inputs, training=training)
    samples_0 = self._sample_channels(*params_0, seed=seed)

    image_height, image_width, _ = tensorshape_util.as_list(self.event_shape)
    def loop_body(index, samples):
      """Loop for iterative pixel sampling.

      Args:
        index: 0D `Tensor` of type `int32`. Index of the current pixel.
        samples: 4D `Tensor`. Images with pixels sampled in raster order, up to
          pixel `[index]`, with dimensions `[batch_size, height, width,
          num_channels]`.

      Returns:
        samples: 4D `Tensor`. Images with pixels sampled in raster order, up to
          and including pixel `[index]`, with dimensions `[batch_size, height,
          width, num_channels]`.
      """
      inputs = samples if conditional_input is None else [samples, h]
      params = self.network(inputs, training=training)
      samples_new = self._sample_channels(*params, seed=seed)

      # Update the current pixel
      samples = tf.transpose(samples, [1, 2, 3, 0])
      samples_new = tf.transpose(samples_new, [1, 2, 3, 0])
      row, col = index // image_width, index % image_width
      updates = samples_new[row, col, Ellipsis][tf.newaxis, Ellipsis]
      samples = tf.tensor_scatter_nd_update(samples, [[row, col]], updates)
      samples = tf.transpose(samples, [3, 0, 1, 2])

      return index + 1, samples

    index0 = tf.zeros([], dtype=tf.int32)

    # Construct the while loop for sampling
    total_pixels = image_height * image_width
    loop_cond = lambda ind, _: tf.less(ind, total_pixels)
    init_vars = (index0, samples_0)
    _, samples = tf.while_loop(
        loop_cond, loop_body, init_vars, parallel_iterations=1)

    if self._rescale_pixel_value:
      transformed_samples = (
          self._low + 0.5 * (self._high - self._low) * (samples + 1.))
      return tf.round(transformed_samples)
    else:
      return tf.round(samples)

  def _sample_channels(
      self, component_logits, locs, scales, coeffs=None, seed=None):
    """Sample a single pixel-iteration and apply channel conditioning.

    Args:
      component_logits: 4D `Tensor` of logits for the Categorical distribution
        over Quantized Logistic mixture components. Dimensions are `[batch_size,
        height, width, num_logistic_mix]`.
      locs: 4D `Tensor` of location parameters for the Quantized Logistic
        mixture components. Dimensions are `[batch_size, height, width,
        num_logistic_mix, num_channels]`.
      scales: 4D `Tensor` of location parameters for the Quantized Logistic
        mixture components. Dimensions are `[batch_size, height, width,
        num_logistic_mix, num_channels]`.
      coeffs: 4D `Tensor` of coefficients for the linear dependence among color
        channels, or `None` if there is only one channel. Dimensions are
        `[batch_size, height, width, num_logistic_mix, num_coeffs]`, where
        `num_coeffs = num_channels * (num_channels - 1) // 2`.
      seed: `int`, random seed.

    Returns:
      samples: 4D `Tensor` of sampled image data with autoregression among
        channels. Dimensions are `[batch_size, height, width, num_channels]`.
    """
    num_channels = self.event_shape[-1]

    # sample mixture components once for the entire pixel
    component_dist = categorical.Categorical(logits=component_logits)
    mask = tf.one_hot(indices=component_dist.sample(seed=seed),
                      depth=self._num_logistic_mix)
    mask = tf.cast(mask[Ellipsis, tf.newaxis], self.dtype)

    # apply mixture component mask and separate out RGB parameters
    masked_locs = tf.reduce_sum(locs * mask, axis=-2)
    loc_tensors = tf.split(masked_locs, num_channels, axis=-1)
    masked_scales = tf.reduce_sum(scales * mask, axis=-2)
    scale_tensors = tf.split(masked_scales, num_channels, axis=-1)

    if coeffs is not None:
      num_coeffs = num_channels * (num_channels - 1) // 2
      masked_coeffs = tf.reduce_sum(coeffs * mask, axis=-2)
      coef_tensors = tf.split(masked_coeffs, num_coeffs, axis=-1)

    channel_samples = []
    coef_count = 0
    for i in range(num_channels):
      loc = loc_tensors[i]
      for c in channel_samples:
        loc += c * coef_tensors[coef_count]
        coef_count += 1

      logistic_samp = logistic.Logistic(
          loc=loc, scale=scale_tensors[i]).sample(seed=seed)
      logistic_samp = tf.clip_by_value(logistic_samp, -1., 1.)
      channel_samples.append(logistic_samp)

    return tf.concat(channel_samples, axis=-1)

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape(self):
    return tf.TensorShape(self.image_shape)


class _PixelCNNNetwork(tf.keras.layers.Layer):
  """Keras `Layer` to parameterize a Pixel CNN++ distribution.

  This is a Keras implementation of the Pixel CNN++ network, as described in
  Salimans et al. (2017)[1] and van den Oord et al. (2016)[2].
  (https://github.com/openai/pixel-cnn).

  #### References

  [1]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.
       PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture
       Likelihood and Other Modifications. In _International Conference on
       Learning Representations_, 2017.
       https://pdfs.semanticscholar.org/9e90/6792f67cbdda7b7777b69284a81044857656.pdf
       Additional details at https://github.com/openai/pixel-cnn

  [2]: Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt,
       Alex Graves, and Koray Kavukcuoglu. Conditional Image Generation with
       PixelCNN Decoders. In _30th Conference on Neural Information Processing
       Systems_, 2016.
       https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf

  """

  def __init__(self,
               dropout_p=0.5,
               num_resnet=5,
               num_hierarchies=3,
               num_filters=160,
               num_logistic_mix=10,
               receptive_field_dims=(3, 3),
               resnet_activation='concat_elu',
               reg_weight=0.0,
               use_weight_norm=True,
               use_data_init=True,
               high=255,
               low=0,
               rescale_pixel_value=True,
               dtype=tf.float32):
    """Initialize the neural network for the Pixel CNN++ distribution.

    Args:
      dropout_p: `float`, the dropout probability. Should be between 0 and 1.
      num_resnet: `int`, the number of layers (shown in Figure 2 of [2]) within
        each highest-level block of Figure 2 of [1].
      num_hierarchies: `int`, the number of hightest-level blocks (separated by
        expansions/contractions of dimensions in Figure 2 of [1].)
      num_filters: `int`, the number of convolutional filters.
      num_logistic_mix: `int`, number of components in the logistic mixture
        distribution.
      receptive_field_dims: `tuple`, height and width in pixels of the receptive
        field of the convolutional layers above and to the left of a given
        pixel. The width (second element of the tuple) should be odd. Figure 1
        (middle) of [2] shows a receptive field of (3, 5) (the row containing
        the current pixel is included in the height). The default of (3, 3) was
        used to produce the results in [1].
      resnet_activation: `string`, the type of activation to use in the resnet
        blocks. May be 'concat_elu', 'elu', or 'relu'.
      reg_weight: 'float', the l2 regularization.
      use_weight_norm: `bool`, if `True` then use weight normalization.
      use_data_init: `bool`, if `True` then use data-dependent initialization
        (has no effect if `use_weight_norm` is `False`).
      high: `int`, the maximum value of the input data (255 for an 8-bit image).
      low: `int`, the minimum value of the input data.
      rescale_pixel_value: 'bool', if `True` then rescale pixel value to [-1,1].
      dtype: Data type of the layer.
    """
    super(_PixelCNNNetwork, self).__init__(dtype=dtype)
    self._dropout_p = dropout_p
    self._num_resnet = num_resnet
    self._num_hierarchies = num_hierarchies
    self._num_filters = num_filters
    self._num_logistic_mix = num_logistic_mix
    self._receptive_field_dims = receptive_field_dims
    self._resnet_activation = resnet_activation
    self._reg_weight = reg_weight
    self._high = high
    self._low = low
    self._rescale_pixel_value = rescale_pixel_value

    if use_weight_norm:
      def layer_wrapper(layer):
        def wrapped_layer(*args, **kwargs):
          return weight_norm.WeightNorm(
              layer(*args, dtype=dtype, **kwargs), data_init=use_data_init)
        return wrapped_layer
      self._layer_wrapper = layer_wrapper
    else:
      self._layer_wrapper = lambda layer: layer

  def build(self, input_shape):
    dtype = self.dtype
    if len(input_shape) == 2:
      batch_image_shape, batch_conditional_shape = input_shape
      conditional_input = tf.keras.layers.Input(
          shape=batch_conditional_shape[1:], dtype=dtype)
    else:
      batch_image_shape = input_shape
      conditional_input = None

    image_shape = batch_image_shape[1:]
    image_input = tf.keras.layers.Input(shape=image_shape, dtype=dtype)

    if self._resnet_activation == 'concat_elu':
      activation = tf.keras.layers.Lambda(
          lambda x: tf.nn.elu(tf.concat([x, -x], axis=-1)), dtype=dtype)
    else:
      activation = tf.keras.activations.get(self._resnet_activation)

    # Define layers with default inputs and layer wrapper applied
    Conv2D = functools.partial(  # pylint:disable=invalid-name
        self._layer_wrapper(tf.keras.layers.Convolution2D),
        filters=self._num_filters,
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(self._reg_weight),
        dtype=dtype)

    Dense = functools.partial(  # pylint:disable=invalid-name
        self._layer_wrapper(tf.keras.layers.Dense),
        kernel_regularizer=tf.keras.regularizers.l2(self._reg_weight),
        dtype=dtype)

    Conv2DTranspose = functools.partial(  # pylint:disable=invalid-name
        self._layer_wrapper(tf.keras.layers.Conv2DTranspose),
        filters=self._num_filters,
        padding='same',
        strides=(2, 2),
        kernel_regularizer=tf.keras.regularizers.l2(self._reg_weight),
        dtype=dtype)

    rows, cols = self._receptive_field_dims

    # Define the dimensions of the valid (unmasked) areas of the layer kernels
    # for stride 1 convolutions in the internal layers.
    kernel_valid_dims = {'vertical': (rows - 1, cols),
                         'horizontal': (2, cols // 2 + 1)}

    # Define the size of the kernel necessary to center the current pixel
    # correctly for stride 1 convolutions in the internal layers.
    kernel_sizes = {'vertical': (2 * rows - 3, cols), 'horizontal': (3, cols)}

    # Make the kernel constraint functions for stride 1 convolutions in internal
    # layers.
    kernel_constraints = {
        k: _make_kernel_constraint(kernel_sizes[k], (0, v[0]), (0, v[1]))
        for k, v in kernel_valid_dims.items()}

    # Build the initial vertical stack/horizontal stack convolutional layers,
    # as shown in Figure 1 of [2]. The receptive field of the initial vertical
    # stack layer is a rectangular area centered above the current pixel.
    vertical_stack_init = Conv2D(
        kernel_size=(2 * rows - 1, cols),
        kernel_constraint=_make_kernel_constraint(
            (2 * rows - 1, cols), (0, rows - 1), (0, cols)))(image_input)

    # In Figure 1 [2], the receptive field of the horizontal stack is
    # illustrated as the pixels in the same row and to the left of the current
    # pixel. [1] increases the height of this receptive field from one pixel to
    # two (`horizontal_stack_left`) and additionally includes a subset of the
    # row of pixels centered above the current pixel (`horizontal_stack_up`).
    horizontal_stack_up = Conv2D(
        kernel_size=(3, cols),
        kernel_constraint=_make_kernel_constraint(
            (3, cols), (0, 1), (0, cols)))(image_input)

    horizontal_stack_left = Conv2D(
        kernel_size=(3, cols),
        kernel_constraint=_make_kernel_constraint(
            (3, cols), (0, 2), (0, cols // 2)))(image_input)

    horizontal_stack_init = tf.keras.layers.add(
        [horizontal_stack_up, horizontal_stack_left], dtype=dtype)

    layer_stacks = {
        'vertical': [vertical_stack_init],
        'horizontal': [horizontal_stack_init]}

    # Build the downward pass of the U-net (left-hand half of Figure 2 of [1]).
    # Each `i` iteration builds one of the highest-level blocks (identified as
    # 'Sequence of 6 layers' in the figure, consisting of `num_resnet=5` stride-
    # 1 layers, and one stride-2 layer that contracts the height/width
    # dimensions). The `_` iterations build the stride 1 layers. The layers of
    # the downward pass are stored in lists, since we'll later need them to make
    # skip-connections to layers in the upward pass of the U-net (the skip-
    # connections are represented by curved lines in Figure 2 [1]).
    for i in range(self._num_hierarchies):
      for _ in range(self._num_resnet):
        # Build a layer shown in Figure 2 of [2]. The 'vertical' iteration
        # builds the layers in the left half of the figure, and the 'horizontal'
        # iteration builds the layers in the right half.
        for stack in ['vertical', 'horizontal']:
          input_x = layer_stacks[stack][-1]
          x = activation(input_x)
          x = Conv2D(kernel_size=kernel_sizes[stack],
                     kernel_constraint=kernel_constraints[stack])(x)

          # Add the vertical-stack layer to the horizontal-stack layer
          if stack == 'horizontal':
            h = activation(layer_stacks['vertical'][-1])
            h = Dense(self._num_filters)(h)
            x = tf.keras.layers.add([h, x], dtype=dtype)

          x = activation(x)
          x = tf.keras.layers.Dropout(self._dropout_p, dtype=dtype)(x)
          x = Conv2D(filters=2*self._num_filters,
                     kernel_size=kernel_sizes[stack],
                     kernel_constraint=kernel_constraints[stack])(x)

          if conditional_input is not None:
            h_projection = _build_and_apply_h_projection(
                conditional_input, self._num_filters, dtype=dtype)
            x = tf.keras.layers.add([x, h_projection], dtype=dtype)

          x = _apply_sigmoid_gating(x)

          # Add a residual connection from the layer's input.
          out = tf.keras.layers.add([input_x, x], dtype=dtype)
          layer_stacks[stack].append(out)

      if i < self._num_hierarchies - 1:
        # Build convolutional layers that contract the height/width dimensions
        # on the downward pass between each set of layers (e.g. contracting from
        # 32x32 to 16x16 in Figure 2 of [1]).
        for stack in ['vertical', 'horizontal']:
          # Define kernel dimensions/masking to maintain the autoregressive
          # property.
          x = layer_stacks[stack][-1]
          h, w = kernel_valid_dims[stack]
          kernel_height = 2 * h
          if stack == 'vertical':
            kernel_width = w + 1
          else:
            kernel_width = 2 * w

          kernel_size = (kernel_height, kernel_width)
          kernel_constraint = _make_kernel_constraint(
              kernel_size, (0, h), (0, w))
          x = Conv2D(strides=(2, 2), kernel_size=kernel_size,
                     kernel_constraint=kernel_constraint)(x)
          layer_stacks[stack].append(x)

    # Upward pass of the U-net (right-hand half of Figure 2 of [1]). We stored
    # the layers of the downward pass in a list, in order to access them to make
    # skip-connections to the upward pass. For the upward pass, we need to keep
    # track of only the current layer, so we maintain a reference to the
    # current layer of the horizontal/vertical stack in the `upward_pass` dict.
    # The upward pass begins with the last layer of the downward pass.
    upward_pass = {key: stack.pop() for key, stack in layer_stacks.items()}

    # As with the downward pass, each `i` iteration builds a highest level block
    # in Figure 2 [1], and the `_` iterations build individual layers within the
    # block.
    for i in range(self._num_hierarchies):
      num_resnet = self._num_resnet if i == 0 else self._num_resnet + 1

      for _ in range(num_resnet):
        # Build a layer as shown in Figure 2 of [2], with a skip-connection
        # from the symmetric layer in the downward pass.
        for stack in ['vertical', 'horizontal']:
          input_x = upward_pass[stack]
          x_symmetric = layer_stacks[stack].pop()

          x = activation(input_x)
          x = Conv2D(kernel_size=kernel_sizes[stack],
                     kernel_constraint=kernel_constraints[stack])(x)

          # Include the vertical-stack layer of the upward pass in the layers
          # to be added to the horizontal layer.
          if stack == 'horizontal':
            x_symmetric = tf.keras.layers.Concatenate(axis=-1, dtype=dtype)(
                [upward_pass['vertical'], x_symmetric])

          # Add a skip-connection from the symmetric layer in the downward
          # pass to the layer `x` in the upward pass.
          h = activation(x_symmetric)
          h = Dense(self._num_filters)(h)
          x = tf.keras.layers.add([h, x], dtype=dtype)

          x = activation(x)
          x = tf.keras.layers.Dropout(self._dropout_p, dtype=dtype)(x)
          x = Conv2D(filters=2*self._num_filters,
                     kernel_size=kernel_sizes[stack],
                     kernel_constraint=kernel_constraints[stack])(x)

          if conditional_input is not None:
            h_projection = _build_and_apply_h_projection(
                conditional_input, self._num_filters, dtype=dtype)
            x = tf.keras.layers.add([x, h_projection], dtype=dtype)

          x = _apply_sigmoid_gating(x)
          upward_pass[stack] = tf.keras.layers.add([input_x, x], dtype=dtype)

    # Define deconvolutional layers that expand height/width dimensions on the
    # upward pass (e.g. expanding from 8x8 to 16x16 in Figure 2 of [1]), with
    # the correct kernel dimensions/masking to maintain the autoregressive
    # property.
      if i < self._num_hierarchies - 1:
        for stack in ['vertical', 'horizontal']:
          h, w = kernel_valid_dims[stack]
          kernel_height = 2 * h - 2
          if stack == 'vertical':
            kernel_width = w + 1
            kernel_constraint = _make_kernel_constraint(
                (kernel_height, kernel_width), (h - 2, kernel_height), (0, w))
          else:
            kernel_width = 2 * w - 2
            kernel_constraint = _make_kernel_constraint(
                (kernel_height, kernel_width), (h - 2, kernel_height),
                (w - 2, kernel_width))

          x = upward_pass[stack]
          x = Conv2DTranspose(kernel_size=(kernel_height, kernel_width),
                              kernel_constraint=kernel_constraint)(x)
          upward_pass[stack] = x

    x_out = tf.keras.layers.ELU(dtype=dtype)(upward_pass['horizontal'])

    # Build final Dense/Reshape layers to output the correct number of
    # parameters per pixel.
    num_channels = tensorshape_util.as_list(image_shape)[-1]
    num_coeffs = num_channels * (num_channels - 1) // 2
    num_out = num_channels * 2 + num_coeffs + 1
    num_out_total = num_out * self._num_logistic_mix
    params = Dense(num_out_total)(x_out)
    params = tf.reshape(params, prefer_static.concat(
        [[-1], image_shape[:-1], [self._num_logistic_mix, num_out]], axis=0))

    # If there is one color channel, split the parameters into a list of three
    # output `Tensor`s: (1) component logits for the Quantized Logistic mixture
    # distribution, (2) location parameters for each component, and (3) scale
    # parameters for each component. If there is more than one color channel,
    # return a fourth `Tensor` for the coefficients for the linear dependence
    # among color channels.
    splits = (3 if num_channels == 1
              else [1, num_channels, num_channels, num_coeffs])
    outputs = tf.split(params, splits, axis=-1)

    # Squeeze singleton dimension from component logits
    outputs[0] = tf.squeeze(outputs[0], axis=-1)

    # Ensure scales are positive and do not collapse to near-zero
    if self._rescale_pixel_value:
      outputs[2] = tf.nn.softplus(outputs[2]) + tf.cast(tf.exp(-7.), self.dtype)
    else:
      outputs[2] = tf.maximum(
          tf.sigmoid(outputs[2]) * self._high, tf.cast(0.25, self.dtype))

    inputs = (
        image_input
        if conditional_input is None else [image_input, conditional_input])
    self._network = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(_PixelCNNNetwork, self).build(input_shape)

  def call(self, inputs, training=None):
    """Call the Pixel CNN network model.

    Args:
      inputs: 4D `Tensor` of image data with dimensions [batch size, height,
        width, channels] or a 2-element `list`. If `list`, the first element is
        the 4D image `Tensor` and the second element is a `Tensor` with
        conditional input data (e.g. VAE encodings or class labels) with the
        same leading batch dimension as the image `Tensor`.
      training: `bool` or `None`. If `bool`, it controls the dropout layer,
        where `True` implies dropout is active. If `None`, it it defaults to
        `tf.keras.backend.learning_phase()`

    Returns:
      outputs: a 3- or 4-element `list` of `Tensor`s in the following order:
        component_logits: 4D `Tensor` of logits for the Categorical distribution
          over Quantized Logistic mixture components. Dimensions are
          `[batch_size, height, width, num_logistic_mix]`.
        locs: 4D `Tensor` of location parameters for the Quantized Logistic
          mixture components. Dimensions are `[batch_size, height, width,
          num_logistic_mix, num_channels]`.
        scales: 4D `Tensor` of location parameters for the Quantized Logistic
          mixture components. Dimensions are `[batch_size, height, width,
          num_logistic_mix, num_channels]`.
        coeffs: 4D `Tensor` of coefficients for the linear dependence among
          color channels, included only if the image has more than one channel.
          Dimensions are `[batch_size, height, width, num_logistic_mix,
          num_coeffs]`, where
          `num_coeffs = num_channels * (num_channels - 1) // 2`.
    """
    return self._network(inputs, training=training)


def _make_kernel_constraint(kernel_size, valid_rows, valid_columns):
  """Make the masking function for layer kernels."""
  mask = np.zeros(kernel_size)
  lower, upper = valid_rows
  left, right = valid_columns
  mask[lower:upper, left:right] = 1.
  mask = mask[:, :, np.newaxis, np.newaxis]
  return lambda x: x * mask


def _build_and_apply_h_projection(h, num_filters, dtype):
  """Project the conditional input."""
  h = tf.keras.layers.Flatten(dtype=dtype)(h)
  h_projection = tf.keras.layers.Dense(
      2*num_filters, kernel_initializer='random_normal', dtype=dtype)(h)
  return h_projection[Ellipsis, tf.newaxis, tf.newaxis, :]


def _apply_sigmoid_gating(x):
  """Apply the sigmoid gating in Figure 2 of [2]."""
  activation_tensor, gate_tensor = tf.split(x, 2, axis=-1)
  sigmoid_gate = tf.sigmoid(gate_tensor)
  return tf.keras.layers.multiply(
      [sigmoid_gate, activation_tensor], dtype=x.dtype)
