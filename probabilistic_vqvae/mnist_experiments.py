# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Trains various vector quantized-variational autoencoder models on MNIST.

See the README.md for experiment details and compilation/running instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import flags
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
from six.moves import urllib
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
import tensorflow as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.training import moving_averages

# TODO(vafa): Set random seed.
IMAGE_SHAPE = [28, 28, 1]
INITIAL_SCALE_BIAS = np.log(np.e / 2. - 1., dtype=np.float32)

flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=10000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("latent_size",
                     default=10,
                     help="Number of latent variables.")
flags.DEFINE_integer("num_codes",
                     default=64,
                     help="Number of discrete codes in codebook.")
flags.DEFINE_integer("code_size",
                     default=16,
                     help="Dimension of each entry in codebook.")
flags.DEFINE_integer("base_depth",
                     default=32,
                     help="Base depth for encoder and decoder CNNs.")
flags.DEFINE_string("activation",
                    default="elu",
                    help="Activation function for all hidden layers.")
flags.DEFINE_float("beta",
                   default=0.25,
                   help="Scaling for commitment loss.")
flags.DEFINE_float("entropy_scale",
                   default=0.,
                   help="Scaling for negative entropy loss.")
flags.DEFINE_float("decay",
                   default=0.99,
                   help="Decay for exponential moving average.")
flags.DEFINE_float("temperature",
                   default=0.5,
                   help="Temperature parameter used for Gumbel-Softmax.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_enum(
    "bottleneck_type",
    default="deterministic",
    enum_values=["deterministic", "categorical", "gumbel_softmax"],
    help="Discrete bottleneck type to be used.")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples for categorical or GS bottleneck.")
flags.DEFINE_integer("num_iaf_flows",
                     default=0,
                     help="Number of IAF flows for bottleneck.")
flags.DEFINE_integer("iaf_startup_steps",
                     default=0,
                     help="Number of startup-steps before applying IAFs.")
flags.DEFINE_boolean("stop_training_encoder_after_startup",
                     default=True,
                     help="Whether to stop training the encoder after startup.")
flags.DEFINE_boolean("use_autoregressive_prior",
                     default=True,
                     help="Whether to use Transformer autoregressive prior.")
flags.DEFINE_boolean("use_transformer_for_iaf_parameters",
                     default=True,
                     help="Whether to use a Transformer instead of a lower-"
                          "triangular mat-mul to generate IAF parameters.")
flags.DEFINE_boolean("sum_over_latents",
                     default=True,
                     help="Whether to sum over latent dimension for training.")
flags.DEFINE_boolean("stop_gradient_for_prior",
                     default=False,
                     help="Whether to stop gradients on prior input.")
flags.DEFINE_boolean("average_categorical_samples",
                     default=True,
                     help="Whether to average categorical samples.")
flags.DEFINE_string("mnist_type",
                    default="threshold",
                    help="""Type of MNIST used. Choices include 'fake_data',
                    'bernoulli' for Hugo Larochelle's randomly binarized MNIST,
                     and 'threshold' for binarized MNIST at 0.5 threshold.""")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "vqvae/data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vqvae/log"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=1000,
                     help="Frequency at which to save visualizations.")
flags.DEFINE_string("master",
                    default="",
                    help="BNS name of the TensorFlow master to use.")

FLAGS = flags.FLAGS
BERNOULLI_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


class MnistType(object):
  """MNIST types for input data."""
  FAKE_DATA = "fake_data"
  THRESHOLD = "threshold"
  BERNOULLI = "bernoulli"


class VectorQuantizer(object):
  """Creates a vector-quantizer.

  It quantizes a continuous vector under a codebook. The codebook is also known
  as "embeddings" or "memory", and it is learned using an exponential moving
  average.
  """

  def __init__(self, num_codes, code_size):
    self.num_codes = num_codes
    self.code_size = code_size
    self.codebook = tf.get_variable(
        "codebook", [num_codes, code_size], dtype=tf.float32,)
    self.ema_count = tf.get_variable(
        name="ema_count", shape=[num_codes],
        initializer=tf.constant_initializer(0), trainable=False)
    self.ema_means = tf.get_variable(
        name="ema_means", initializer=self.codebook.initialized_value(),
        trainable=False)

  def __call__(self, codes):
    """Uses codebook to find nearest neighbor for each code.

    Args:
      codes: A `float`-like `Tensor` containing the latent
        vectors to be compared to the codebook. These are rank-3 with shape
        `[batch_size, latent_size, code_size]`.

    Returns:
      nearest_codebook_entries: The 1-nearest neighbor in Euclidean distance for
        each code in the batch.
      one_hot_assignments: The one-hot vectors corresponding to the matched
        codebook entry for each code in the batch.
      distances: The Euclidean distances between for each code.
    """
    distances = tf.norm(
        codes[:, :, tf.newaxis, ...] -
        tf.reshape(self.codebook, [1, 1, self.num_codes, self.code_size]),
        axis=3)
    assignments = tf.argmin(distances, 2)
    one_hot_assignments = tf.one_hot(assignments, depth=self.num_codes)
    nearest_codebook_entries = tf.reduce_sum(
        one_hot_assignments[..., tf.newaxis] *
        tf.reshape(self.codebook, [1, 1, self.num_codes, self.code_size]),
        axis=2)
    return nearest_codebook_entries, one_hot_assignments, distances


class Encoder(tf.keras.Model):
  """Convolutional neural net that outputs latents codes."""

  def __init__(self, base_depth, activation, latent_size, code_size):
    """Creates the encoder function.

    Args:
      base_depth: Layer base depth in encoder net.
      activation: Activation function in hidden layers.
      latent_size: The number of latent variables in the code.
      code_size: The dimensionality of each latent variable.
    """
    super(Encoder, self).__init__()
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    self.encoder_net = tf.keras.Sequential([
        conv(base_depth, 5, 1),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 1),
        conv(2 * base_depth, 5, 2),
        conv(4 * latent_size, 7, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_size * code_size, activation=None),
        tf.keras.layers.Reshape([latent_size, code_size])
    ])

  def call(self, images):
    """Encodes a batch of images.

    Args:
      images: A `Tensor` representing the inputs to be encoded, of shape `[...,
        channels]`.

    Returns:
      codes: A `float`-like `Tensor` of shape `[..., latent_size, code_size]`.
        It represents latent vectors to be matched with the codebook.
    """
    images = 2 * tf.cast(images, dtype=tf.float32) - 1
    codes = self.encoder_net(images)
    return codes


class Decoder(tf.keras.Model):
  """Convolutional neural net that parameterizes Bernoulli distribution."""

  def __init__(self, base_depth, activation, input_size, image_shape):
    """Creates the decoder function.

    Args:
      base_depth: Layer base depth in decoder net.
      activation: Activation function in hidden layers.
      input_size: The flattened latent input shape as an int.
      image_shape: The output image shape as a list.
    """
    super(Decoder, self).__init__()
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)
    self.decoder_net = tf.keras.Sequential([
        tf.keras.layers.Reshape((1, 1, input_size)),
        deconv(2 * base_depth, 7, padding="VALID"),
        deconv(2 * base_depth, 5),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5),
        conv(image_shape[-1], 5, activation=None),
        tf.keras.layers.Reshape(image_shape),
    ])
    self.image_shape = image_shape

  def call(self, codes):
    """Builds a distribution over images given codes.

    Args:
      codes: A `Tensor` representing the inputs to be decoded, of shape `[...,
        code_size]`.

    Returns:
      decoder_distribution: A multivariate `Bernoulli` distribution.
    """
    num_samples, batch_size, latent_size, code_size = common_layers.shape_list(
        codes)
    codes = tf.reshape(codes,
                       [num_samples * batch_size, latent_size, code_size])
    logits = self.decoder_net(codes)
    logits = tf.reshape(logits,
                        [num_samples, batch_size] + list(self.image_shape))
    return tfd.Independent(tfd.Bernoulli(logits=logits),
                           reinterpreted_batch_ndims=len(self.image_shape),
                           name="decoder_distribution")


def transformer_hparams(hidden_size):
  """Creates hyperpameters for autoregressive prior.

  Args:
    hidden_size: Width of attention layers and neural network output layer.

  Returns:
    hparams: Hyperpameters with basic presets for a Transformer.
  """
  hparams = transformer.transformer_tiny()
  hparams.add_hparam("shared_rel", False)
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)
  hparams.hidden_size = hidden_size
  hparams.num_layers = 6
  hparams.layer_prepostprocess_dropout = 0.
  hparams.attention_dropout = 0.
  hparams.relu_dropout = 0.
  hparams.block_length = 1
  hparams.block_width = 1
  hparams.ffn_layer = "conv_hidden_relu"
  return hparams


def make_uniform_prior():
  """Creates uniform prior."""
  def prior_fn(codes):
    logits = tf.zeros_like(codes)
    prior_dist = tfd.OneHotCategorical(logits=logits, dtype=tf.float32)
    return prior_dist
  return prior_fn


def make_transformer_prior(num_codes, code_size):
  """Creates autoregressive prior using a Transformer.

  Args:
    num_codes: Number of codes in codebook.
    code_size: Dimension of each code in codebook.

  Returns:
    prior_fn: A callable mapping of a Tensor of discete latents of shape
      [num_samples, batch_size, latent_size, num_codes] to a Tensor of the same
      shape.
  """

  embedding_layer = tf.get_variable(
      "prior_embedding", [num_codes, code_size], dtype=tf.float32)
  hparams = transformer_hparams(hidden_size=code_size)

  def prior_fn(shifted_codes):
    """Calculates prior logits on discrete latents.

    Args:
      shifted_codes: A binary `Tensor` of shape
        [num_samples, batch_size, latent_size, num_codes], shifted by one to
        enable autoregressive calculation.

    Returns:
      prior_dist: Multinomial distribution with prior logits coming from
        Transformer applied to shifted input.
    """
    with tf.variable_scope("transformer_prior", reuse=tf.AUTO_REUSE):
      dense_shifted_codes = tf.reduce_sum(
          tf.reshape(embedding_layer, [1, 1, 1, num_codes, code_size]) *
          shifted_codes[..., tf.newaxis], axis=-2)
      transformed_codes = cia.transformer_decoder_layers(
          inputs=dense_shifted_codes,
          encoder_output=None,
          num_layers=hparams.num_layers,
          hparams=hparams,
          attention_type=cia.AttentionType.LOCAL_1D)
      logits = tf.reduce_sum(
          tf.reshape(embedding_layer, [1, 1, 1, num_codes, code_size]) *
          transformed_codes[..., tf.newaxis, :], axis=-1)
      prior_dist = tfd.Multinomial(total_count=1., logits=logits)
    return prior_dist

  return prior_fn


def make_prior_loss(prior_fn, one_hot_assignments, sum_over_latents):
  """Computes prior loss from latent assignments.

  Args:
   prior_fn: A callable to calculate the logits for the latent prior.
   one_hot_assignments: The one-hot vectors corresponding to the matched
      codebook entry for each code in the batch. Should be [num_samples,
      batch_size, latent_size, num_codes].
   sum_over_latents: Whether to sum over latent dimension when computing loss.

  Returns:
    prior_loss: Scalar loss, scaled by either 1/batch_size or 1/(batch_size *
      latent_size).
  """
  padded_codes = shift_assignments(one_hot_assignments)
  prior_dist = prior_fn(padded_codes)
  prior_loss = prior_dist.log_prob(one_hot_assignments)
  if sum_over_latents:
    prior_loss = tf.reduce_sum(prior_loss, axis=-1)
  prior_loss = -tf.reduce_mean(prior_loss)
  return prior_loss


def categorical_bottleneck(dist,
                           vector_quantizer,
                           num_iaf_flows=0,
                           average_categorical_samples=True,
                           use_transformer_for_iaf_parameters=False,
                           sum_over_latents=True,
                           num_samples=1,
                           summary=True):
  """Implements soft EM bottleneck using averaged categorical samples.

  Args:
    dist: Distances between encoder outputs and codebook entries. Negative
      distances are used as categorical logits. A float Tensor of shape
      [batch_size, latent_size, code_size].
    vector_quantizer: An instance of the VectorQuantizer class.
    num_iaf_flows: Number of inverse autoregressive flows.
    average_categorical_samples: Whether to take the average of `num_samples'
      categorical samples as in Roy et al. or to use `num_samples` categorical
      samples to approximate gradient.
    use_transformer_for_iaf_parameters: Whether to use a Transformer instead of
      a lower-triangular mat-mul for generating IAF parameters.
    sum_over_latents: Whether to sum over latent dimension when computing
      entropy.
    num_samples: Number of categorical samples.
    summary: Whether to log entropy histogram.

  Returns:
    one_hot_assignments: Simplex-valued assignments sampled from categorical.
    neg_q_entropy: Negative entropy of categorical distribution.
  """
  latent_size = dist.shape[1]
  x_means_idx = tf.multinomial(
      logits=tf.reshape(-dist, [-1, FLAGS.num_codes]),
      num_samples=num_samples)
  one_hot_assignments = tf.one_hot(x_means_idx,
                                   depth=vector_quantizer.num_codes)
  one_hot_assignments = tf.reshape(
      one_hot_assignments,
      [-1, latent_size, num_samples, vector_quantizer.num_codes])
  if average_categorical_samples:
    summed_assignments = tf.reduce_sum(one_hot_assignments, axis=2)
    averaged_assignments = tf.reduce_mean(one_hot_assignments, axis=2)
    entropy_dist = tfd.Multinomial(
        total_count=tf.cast(num_samples, tf.float32),
        logits=-dist)
    neg_q_entropy = entropy_dist.log_prob(summed_assignments)
    one_hot_assignments = averaged_assignments[tf.newaxis, ...]
  else:
    one_hot_assignments = tf.transpose(one_hot_assignments, [2, 0, 1, 3])
    entropy_dist = tfd.OneHotCategorical(logits=-dist, dtype=tf.float32)
    neg_q_entropy = entropy_dist.log_prob(one_hot_assignments)

  if summary:
    tf.summary.histogram("neg_q_entropy_0",
                         tf.reshape(tf.reduce_sum(neg_q_entropy, axis=-1),
                                    [-1]))
  # Perform straight-through.
  class_probs = tf.nn.softmax(-dist[tf.newaxis, ...])
  one_hot_assignments = class_probs + tf.stop_gradient(
      one_hot_assignments - class_probs)

  # Perform IAF flows
  for flow_num in range(num_iaf_flows):
    with tf.variable_scope("iaf_variables", reuse=tf.AUTO_REUSE):
      # Pad the one_hot_assignments by zeroing out the first latent dimension
      # and shifting the rest down by one (and removing the last dimension).
      shifted_codes = shift_assignments(one_hot_assignments)
      if use_transformer_for_iaf_parameters:
        unconstrained_scale = iaf_scale_from_transformer(
            shifted_codes,
            vector_quantizer.code_size,
            name=str(flow_num))
      else:
        unconstrained_scale = iaf_scale_from_matmul(shifted_codes,
                                                    name=str(flow_num))
      # Initialize scale bias to be log(e/2 - 1) so initial scale + scale_bias
      # is 1.
      initial_scale_bias = tf.fill([latent_size, vector_quantizer.num_codes],
                                   INITIAL_SCALE_BIAS)
      scale_bias = tf.get_variable("scale_bias_" + str(flow_num),
                                   initializer=initial_scale_bias)

    # Since categorical is discrete, we don't need to add inverse log
    # determinant.
    one_hot_assignments, _ = iaf_flow(one_hot_assignments,
                                      unconstrained_scale,
                                      scale_bias,
                                      summary=summary)

  if sum_over_latents:
    neg_q_entropy = tf.reduce_sum(neg_q_entropy, axis=-1)
  neg_q_entropy = tf.reduce_mean(neg_q_entropy)

  return one_hot_assignments, neg_q_entropy


def iaf_scale_from_transformer(shifted_codes,
                               code_size,
                               name=None):
  """Returns unconstrained IAF scale tensor generated by Transformer.

  Args:
    shifted_codes: Tensor with shape [num_samples, batch_size, latent_size,
      num_codes], with the first latent_size dimension a tensor of zeros and the
      others shifted down one (with the original last latent dimension missing).
    code_size: Size of each latent code.
    name: String used for name scope.

  Returns:
    unconstrained_scale: Tensor with shape [latent_size, latent_size],
      generated by passing shifted codes through Transformer decoder layers. It
      can take on any real value as it will later be passed through a softplus.
  """
  with tf.name_scope(name, default_name="iaf_scale_from_transformer") as name:
    num_codes = shifted_codes.shape[-1]
    embedding_layer = tf.get_variable(name + "iaf_embedding",
                                      [num_codes, code_size],
                                      dtype=tf.float32)
    hparams = transformer_hparams(hidden_size=code_size)
    dense_shifted_codes = tf.reduce_sum(
        tf.reshape(embedding_layer, [1, 1, 1, num_codes, code_size]) *
        shifted_codes[..., tf.newaxis], axis=-2)
    transformed_codes = cia.transformer_decoder_layers(
        inputs=dense_shifted_codes,
        encoder_output=None,
        num_layers=hparams.num_layers,
        hparams=hparams,
        attention_type=cia.AttentionType.LOCAL_1D,
        name=name)
    unconstrained_scale = tf.reduce_sum(
        tf.reshape(embedding_layer, [1, 1, 1, num_codes, code_size]) *
        transformed_codes[..., tf.newaxis, :], axis=-1)
    # Multiplying by scale_weight allows identity initialization.
    scale_weight = tf.get_variable(name + "scale_weight",
                                   [],
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
    unconstrained_scale = unconstrained_scale * scale_weight
    return unconstrained_scale


def iaf_scale_from_matmul(shifted_codes, name=None):
  """Returns IAF scale matrix generated by lower-triangular mat-mul.

  Args:
    shifted_codes: Tensor with shape [num_samples, batch_size, latent_size,
      num_codes], with the first latent_size dimension a tensor of zeros and the
      others shifted down one (with the original last latent dimension missing).
    name: String used for name scope.

  Returns:
    unconstrained_scale: Tensor with shape [latent_size, latent_size],
      generated by multiplying padded codes by lower-triangular matrix. It can
      take on any real value as it will later be passed through a softplus.
  """
  with tf.name_scope(name, default_name="iaf_scale_from_transformer") as name:
    latent_size = int(shifted_codes.shape[2])
    scale_matrix = tf.get_variable(name + "scale_matrix",
                                   [latent_size * (latent_size + 1) / 2],
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
    scale_bijector = tfb.Affine(
        scale_tril=tfd.fill_triangular(scale_matrix))
    unconstrained_scale = scale_bijector.forward(
        tf.transpose(shifted_codes, [0, 1, 3, 2]))
    # Transpose the bijector output since it performs a batch matmul.
    unconstrained_scale = tf.transpose(unconstrained_scale, [0, 1, 3, 2])
    return unconstrained_scale


def gumbel_softmax_bottleneck(dist,
                              vector_quantizer,
                              temperature=0.5,
                              num_iaf_flows=0,
                              use_transformer_for_iaf_parameters=False,
                              num_samples=1,
                              sum_over_latents=True,
                              summary=True):
  """Gumbel-Softmax discrete bottleneck.

  Args:
    dist: Distances between encoder outputs and codebook entries, to be used as
      categorical logits. A float Tensor of shape [batch_size, latent_size,
      code_size].
    vector_quantizer: An instance of the VectorQuantizer class.
    temperature: Temperature parameter used for Gumbel-Softmax distribution.
    num_iaf_flows: Number of inverse-autoregressive flows to perform.
    use_transformer_for_iaf_parameters: Whether to use a Transformer instead of
      a lower-triangular mat-mul to generate IAF parameters.
    num_samples: Number of categorical samples.
    sum_over_latents: Whether to sum over latent dimension when computing
      entropy.
    summary: Whether to log summary histogram.

  Returns:
    one_hot_assignments: Simplex-valued assignments sampled from categorical.
    neg_q_entropy: Negative entropy of categorical distribution.
  """
  latent_size = dist.shape[1]
  # TODO(vafa): Consider randomly setting high temperature to help training.
  one_hot_assignments = tfd.RelaxedOneHotCategorical(
      temperature=temperature,
      logits=-dist).sample(num_samples)
  one_hot_assignments = tf.clip_by_value(one_hot_assignments, 1e-6, 1-1e-6)

  # Approximate density with multinomial distribution.
  q_dist = tfd.Multinomial(total_count=1., logits=-dist)
  neg_q_entropy = q_dist.log_prob(one_hot_assignments)
  if summary:
    tf.summary.histogram("neg_q_entropy_0",
                         tf.reshape(tf.reduce_sum(neg_q_entropy, axis=-1),
                                    [-1]))

  # Perform IAF flows
  for flow_num in range(num_iaf_flows):
    with tf.variable_scope("iaf_variables", reuse=tf.AUTO_REUSE):
      # Pad the one_hot_assignments by zeroing out the first latent dimension
      # and shifting the rest down by one (and removing the last dimension).
      shifted_codes = shift_assignments(one_hot_assignments)
      if use_transformer_for_iaf_parameters:
        unconstrained_scale = iaf_scale_from_transformer(
            shifted_codes,
            vector_quantizer.code_size,
            name=str(flow_num))
      else:
        unconstrained_scale = iaf_scale_from_matmul(shifted_codes,
                                                    name=str(flow_num))
      # Initialize scale bias to be log(e/2 - 1) so initial scale + scale_bias
      # evaluates to 1.
      initial_scale_bias = tf.fill([latent_size, vector_quantizer.num_codes],
                                   INITIAL_SCALE_BIAS)
      scale_bias = tf.get_variable("scale_bias_" + str(flow_num),
                                   initializer=initial_scale_bias)

    one_hot_assignments, inverse_log_det_jacobian = iaf_flow(
        one_hot_assignments,
        unconstrained_scale,
        scale_bias,
        summary=summary)
    neg_q_entropy += inverse_log_det_jacobian

  if sum_over_latents:
    neg_q_entropy = tf.reduce_sum(neg_q_entropy, axis=-1)
  neg_q_entropy = tf.reduce_mean(neg_q_entropy)
  return one_hot_assignments, neg_q_entropy


def iaf_flow(one_hot_assignments,
             unconstrained_scale,
             scale_bias,
             summary=True,
             name=None):
  """Performs a single IAF flow using scale and normalization transformations.

  Args:
    one_hot_assignments: Assignments Tensor with shape [num_samples, batch_size,
      latent_size, num_codes].
    unconstrained_scale: Tensor corresponding to scale matrix, generated via
      an autoregressive transformation. This tensor is initially unconstrained
      and will later be passed through softplus.
    scale_bias: Bias tensor to be added to scale tensor, with shape
      [latent_size, num_codes]. If scale weights are zero, initialize scale_bias
      to be log(exp(1.) / 2. - 1) so initial transformation is identity.
    summary: Whether to save summaries.
    name: String used for name scope.

  Returns:
    flow_output: Transformed one-hot assignments.
    inverse_log_det_jacobian: Inverse log deteriminant of Jacobian corresponding
      to transformation.
  """
  with tf.name_scope(name, default_name="iaf"):
    scale = tf.nn.softplus(unconstrained_scale)
    # Add scale bias so we can initialize to identity.
    scale = scale + tf.nn.softplus(scale_bias[tf.newaxis, tf.newaxis, ...])
    scale = scale[..., :-1]

    z = one_hot_assignments[..., :-1]
    unnormalized_probs = tf.concat([z * scale,
                                    one_hot_assignments[..., -1, tf.newaxis]],
                                   axis=-1)
    normalizer = tf.reduce_sum(unnormalized_probs, axis=-1)
    flow_output = unnormalized_probs / (normalizer[..., tf.newaxis])

    num_codes = tf.cast(one_hot_assignments.shape[-1], tf.float32)
    inverse_log_det_jacobian = (-tf.reduce_sum(tf.log(scale), axis=-1)
                                + num_codes * tf.log(normalizer))
    if summary:
      tf.summary.histogram("scale", tf.reshape(scale, [-1]))
      tf.summary.histogram("inverse_log_det_jacobian",
                           tf.reshape(inverse_log_det_jacobian, [-1]))
    return flow_output, inverse_log_det_jacobian


def add_ema_control_dependencies(vector_quantizer,
                                 one_hot_assignments,
                                 codes,
                                 commitment_loss,
                                 decay):
  """Adds control dependencies to the commmitment loss to update the codebook.

  Args:
    vector_quantizer: An instance of the VectorQuantizer class.
    one_hot_assignments: The one-hot vectors corresponding to the matched
      codebook entry for each code in the batch.
    codes: A `float`-like `Tensor` containing the latent vectors to be compared
      to the codebook.
    commitment_loss: The commitment loss from comparing the encoder outputs to
      their neighboring codebook entries.
    decay: Decay factor for exponential moving average.

  Returns:
    commitment_loss: Commitment loss with control dependencies.
  """
  # Use an exponential moving average to update the codebook.
  updated_ema_count = moving_averages.assign_moving_average(
      vector_quantizer.ema_count, tf.reduce_sum(
          one_hot_assignments, axis=[0, 1]), decay, zero_debias=False)
  updated_ema_means = moving_averages.assign_moving_average(
      vector_quantizer.ema_means, tf.reduce_sum(
          codes[:, :, tf.newaxis, ...] *
          one_hot_assignments[..., tf.newaxis], axis=[0, 1]),
      decay, zero_debias=False)

  # Add small value to avoid dividing by zero.
  updated_ema_count += 1e-5
  updated_ema_means /= updated_ema_count[..., tf.newaxis]
  with tf.control_dependencies([commitment_loss]):
    update_means = tf.assign(vector_quantizer.codebook, updated_ema_means)
    with tf.control_dependencies([update_means]):
      return tf.identity(commitment_loss)


def shift_assignments(one_hot_assignments):
  """Returns shifted assignments so predictions can be made autoregressively.

  Args:
    one_hot_assignments: A tensor with shape [num_samples, batch_size,
      latent_size, num_codes] containing one hot assignments to discrete groups.

  Returns:
    shifted_assignments: A tensor with the same shape as one_hot_assignments,
      padded by zeroing out the first latent dimension and shifting the rest
      down by one.
  """
  shifted_assignments = tf.pad(
      one_hot_assignments, [[0, 0], [0, 0], [1, 0], [0, 0]])[:, :, :-1, :]
  return shifted_assignments


def save_imgs(x, fname):
  """Helper method to save a grid of images to a PNG file.

  Args:
    x: A numpy array of shape [n_images, height, width].
    fname: The filename to write to (including extension).
  """
  n = x.shape[0]
  fig = figure.Figure(figsize=(n, 1), frameon=False)
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(1, n, i+1)
    ax.imshow(x[i].squeeze(),
              interpolation="none",
              cmap=cm.get_cmap("binary"))
    ax.axis("off")
  canvas.print_figure(fname, format="png")
  print("saved %s" % fname)


def visualize_training(images_val,
                       reconstructed_images_val,
                       random_images_val,
                       log_dir, prefix, viz_n=10):
  """Helper method to save images visualizing model reconstructions.

  Args:
    images_val: Numpy array containing a batch of input images.
    reconstructed_images_val: Numpy array giving the expected output
      (mean) of the decoder.
    random_images_val: Optionally, a Numpy array giving the expected output
      (mean) of decoding samples from the prior, or `None`.
    log_dir: The directory to write images (Python `str`).
    prefix: A specific label for the saved visualizations, which
      determines their filenames (Python `str`).
    viz_n: The number of images from each batch to visualize (Python `int`).
  """
  save_imgs(images_val[:viz_n],
            os.path.join(log_dir, "{}_inputs.png".format(prefix)))
  save_imgs(reconstructed_images_val[:viz_n],
            os.path.join(log_dir,
                         "{}_reconstructions.png".format(prefix)))

  if random_images_val is not None:
    save_imgs(random_images_val[:viz_n],
              os.path.join(log_dir,
                           "{}_prior_samples.png".format(prefix)))


def build_fake_data(num_examples=10):
  """Builds fake MNIST-style data for unit testing."""

  class Dummy(object):
    pass

  num_examples = 10
  mnist_data = Dummy()
  mnist_data.train = Dummy()
  mnist_data.train.images = np.float32(np.random.randn(
      num_examples, np.prod(IMAGE_SHAPE)))
  mnist_data.train.labels = np.int32(np.random.permutation(
      np.arange(num_examples)))
  mnist_data.train.num_examples = num_examples
  mnist_data.validation = Dummy()
  mnist_data.validation.images = np.float32(np.random.randn(
      num_examples, np.prod(IMAGE_SHAPE)))
  mnist_data.validation.labels = np.int32(np.random.permutation(
      np.arange(num_examples)))
  mnist_data.validation.num_examples = num_examples
  return mnist_data


def download(directory, filename):
  """Downloads a file."""
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  url = os.path.join(BERNOULLI_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath


def load_bernoulli_mnist_dataset(directory, split_name):
  """Returns Hugo Larochelle's binary static MNIST tf.data.Dataset."""
  amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
  dataset = tf.data.TextLineDataset(amat_file)
  str_to_arr = lambda string: np.array([c == b"1" for c in string.split()])

  def _parser(s):
    booltensor = tf.py_func(str_to_arr, [s], tf.bool)
    reshaped = tf.reshape(booltensor, [28, 28, 1])
    return tf.to_float(reshaped), tf.constant(0, tf.int32)

  return dataset.map(_parser)


def build_input_pipeline(data_dir, batch_size, heldout_size, mnist_type):
  """Builds an Iterator switching between train and heldout data."""
  # Build an iterator over training batches.
  if mnist_type in [MnistType.FAKE_DATA, MnistType.THRESHOLD]:
    if mnist_type == MnistType.FAKE_DATA:
      mnist_data = build_fake_data()
    else:
      mnist_data = mnist.read_data_sets(data_dir)
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.train.images, np.int32(mnist_data.train.labels)))
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.validation.images,
         np.int32(mnist_data.validation.labels)))
  elif mnist_type == MnistType.BERNOULLI:
    training_dataset = load_bernoulli_mnist_dataset(data_dir, "train")
    heldout_dataset = load_bernoulli_mnist_dataset(data_dir, "valid")
  else:
    raise ValueError("Unknown MNIST type.")

  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  # TODO(vafa): Consider changing heldout size to a small, fixed amount.
  heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
  heldout_iterator = heldout_frozen.make_one_shot_iterator()

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()
  # Reshape as a pixel image and binarize pixels.
  images = tf.reshape(images, shape=[-1] + IMAGE_SHAPE)
  if mnist_type in [MnistType.FAKE_DATA, MnistType.THRESHOLD]:
    images = tf.cast(images > 0.5, dtype=tf.int32)

  return images, labels, handle, training_iterator, heldout_iterator


def main(argv):
  del argv  # unused
  FLAGS.activation = getattr(tf.nn, FLAGS.activation)
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # TODO(b/113163167): Speed up and tune hyperparameters for Bernoulli MNIST.
    (images, _, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         FLAGS.data_dir, FLAGS.batch_size, heldout_size=10000,
         mnist_type=FLAGS.mnist_type)

    encoder = Encoder(FLAGS.base_depth,
                      FLAGS.activation,
                      FLAGS.latent_size,
                      FLAGS.code_size)
    decoder = Decoder(FLAGS.base_depth,
                      FLAGS.activation,
                      FLAGS.latent_size * FLAGS.code_size,
                      IMAGE_SHAPE)
    vector_quantizer = VectorQuantizer(FLAGS.num_codes, FLAGS.code_size)

    codes = encoder(images)
    nearest_codebook_entries, one_hot_assignments, dist = vector_quantizer(
        codes)
    if FLAGS.bottleneck_type == "deterministic":
      one_hot_assignments = one_hot_assignments[tf.newaxis, ...]
      neg_q_entropy = 0.
      # Perform straight-through.
      class_probs = tf.nn.softmax(-dist[tf.newaxis, ...])
      one_hot_assignments = class_probs + tf.stop_gradient(
          one_hot_assignments - class_probs)
    elif FLAGS.bottleneck_type == "categorical":
      one_hot_assignments, neg_q_entropy = (
          categorical_bottleneck(dist,
                                 vector_quantizer,
                                 FLAGS.num_iaf_flows,
                                 FLAGS.average_categorical_samples,
                                 FLAGS.use_transformer_for_iaf_parameters,
                                 FLAGS.sum_over_latents,
                                 FLAGS.num_samples))
    elif FLAGS.bottleneck_type == "gumbel_softmax":
      one_hot_assignments, neg_q_entropy = (
          gumbel_softmax_bottleneck(dist,
                                    vector_quantizer,
                                    FLAGS.temperature,
                                    FLAGS.num_iaf_flows,
                                    FLAGS.use_transformer_for_iaf_parameters,
                                    FLAGS.num_samples,
                                    FLAGS.sum_over_latents,
                                    summary=True))
    else:
      raise ValueError("Unknown bottleneck type.")

    bottleneck_output = tf.reduce_sum(
        one_hot_assignments[..., tf.newaxis] *
        tf.reshape(
            vector_quantizer.codebook,
            [1, 1, 1, FLAGS.num_codes, FLAGS.code_size]),
        axis=3)

    decoder_distribution = decoder(bottleneck_output)
    reconstructed_images = decoder_distribution.mean()[0]  # get first sample

    reconstruction_loss = -tf.reduce_mean(decoder_distribution.log_prob(images))
    commitment_loss = tf.reduce_mean(
        tf.square(codes[tf.newaxis, ...] -
                  tf.stop_gradient(nearest_codebook_entries)))
    commitment_loss = add_ema_control_dependencies(
        vector_quantizer,
        tf.reduce_mean(one_hot_assignments, axis=0),  # reduce mean over samples
        codes,
        commitment_loss,
        FLAGS.decay)

    if FLAGS.use_autoregressive_prior:
      prior_fn = make_transformer_prior(FLAGS.num_codes, FLAGS.code_size)
    else:
      prior_fn = make_uniform_prior()
    prior_inputs = one_hot_assignments
    if FLAGS.stop_gradient_for_prior:
      prior_inputs = tf.stop_gradient(one_hot_assignments)
    prior_loss = make_prior_loss(prior_fn,
                                 prior_inputs,
                                 sum_over_latents=FLAGS.sum_over_latents)

    loss = (reconstruction_loss + FLAGS.beta * commitment_loss + prior_loss +
            FLAGS.entropy_scale * neg_q_entropy)

    if FLAGS.bottleneck_type == "deterministic":
      if not FLAGS.sum_over_latents:
        prior_loss = prior_loss * FLAGS.latent_size
      heldout_prior_loss = prior_loss
      heldout_reconstruction_loss = reconstruction_loss
      heldout_neg_q_entropy = tf.constant(0.)
    if FLAGS.bottleneck_type == "categorical":
      # To accurately evaluate heldout NLL, we need to sum over latent dimension
      # and use a single sample for the categorical (and not multinomial) prior.
      (heldout_one_hot_assignments,
       heldout_neg_q_entropy) = categorical_bottleneck(
           dist,
           vector_quantizer,
           FLAGS.num_iaf_flows,
           FLAGS.average_categorical_samples,
           FLAGS.use_transformer_for_iaf_parameters,
           sum_over_latents=True,
           num_samples=1,
           summary=False)
      heldout_bottleneck_output = tf.reduce_sum(
          heldout_one_hot_assignments[..., tf.newaxis] *
          tf.reshape(
              vector_quantizer.codebook,
              [1, 1, 1, FLAGS.num_codes, FLAGS.code_size]),
          axis=3)
      heldout_prior_loss = make_prior_loss(
          prior_fn,
          heldout_one_hot_assignments,
          sum_over_latents=True)
      heldout_decoder_distribution = decoder(heldout_bottleneck_output)
      heldout_reconstruction_loss = -tf.reduce_mean(
          heldout_decoder_distribution.log_prob(images))
    elif FLAGS.bottleneck_type == "gumbel_softmax":
      num_test_samples = 1
      heldout_q_dist = tfd.OneHotCategorical(logits=-dist, dtype=tf.float32)
      heldout_one_hot_assignments = heldout_q_dist.sample(num_test_samples)
      heldout_neg_q_entropy = heldout_q_dist.log_prob(
          heldout_one_hot_assignments)
      for flow_num in range(FLAGS.num_iaf_flows):
        with tf.variable_scope("iaf_variables", reuse=tf.AUTO_REUSE):
          shifted_codes = shift_assignments(heldout_one_hot_assignments)
          scale_bias = tf.get_variable("scale_bias_" + str(flow_num))
          if FLAGS.use_transformer_for_iaf_parameters:
            unconstrained_scale = iaf_scale_from_transformer(
                shifted_codes,
                FLAGS.code_size,
                name=str(flow_num))
          else:
            unconstrained_scale = iaf_scale_from_matmul(shifted_codes,
                                                        name=str(flow_num))
        # Don't need to add inverse log determinant jacobian since samples are
        # discrete (no change in volume when bijecting discrete variables).
        heldout_one_hot_assignments, _ = iaf_flow(heldout_one_hot_assignments,
                                                  unconstrained_scale,
                                                  scale_bias,
                                                  summary=False)
      heldout_neg_q_entropy = tf.reduce_sum(
          tf.reshape(heldout_neg_q_entropy, [-1, FLAGS.latent_size]), axis=1)
      heldout_neg_q_entropy = tf.reduce_mean(heldout_neg_q_entropy)
      heldout_nearest_codebook_entries = tf.reduce_sum(
          heldout_one_hot_assignments[..., tf.newaxis] *
          tf.reshape(
              vector_quantizer.codebook,
              [1, 1, 1, FLAGS.num_codes, FLAGS.code_size]),
          axis=3)
      # We still evaluate the prior on the transformed samples. But in order
      # for this categorical distribution to be valid, we have to binarize.
      heldout_one_hot_assignments = tf.one_hot(
          tf.argmax(heldout_one_hot_assignments, axis=-1),
          depth=FLAGS.num_codes)

      heldout_prior_loss = make_prior_loss(
          prior_fn,
          heldout_one_hot_assignments,
          sum_over_latents=True)
      heldout_decoder_distribution = decoder(heldout_nearest_codebook_entries)
      heldout_reconstruction_loss = -tf.reduce_mean(
          heldout_decoder_distribution.log_prob(images))

    marginal_nll = (heldout_prior_loss + heldout_reconstruction_loss +
                    heldout_neg_q_entropy)

    tf.summary.scalar("losses/total_loss", loss)
    tf.summary.scalar("losses/neg_q_entropy_loss",
                      neg_q_entropy * FLAGS.entropy_scale)
    tf.summary.scalar("losses/reconstruction_loss", reconstruction_loss)
    tf.summary.scalar("losses/prior_loss", prior_loss)
    tf.summary.scalar("losses/commitment_loss", FLAGS.beta * commitment_loss)

    tf.summary.scalar("heldout/neg_q_entropy_loss",
                      heldout_neg_q_entropy,
                      collections=["heldout"])
    tf.summary.scalar("heldout/reconstruction_loss",
                      heldout_reconstruction_loss,
                      collections=["heldout"])
    tf.summary.scalar("heldout/prior_loss",
                      heldout_prior_loss,
                      collections=["heldout"])

    # Decode 10 samples from prior for visualization.
    if FLAGS.use_autoregressive_prior:
      assignments = tf.zeros([10, 1, FLAGS.num_codes])
      # Decode autoregressively.
      for d in range(FLAGS.latent_size):
        logits = prior_fn(assignments).logits
        latent_dim_logit = logits[0, :, tf.newaxis, d, :]
        sample = tfd.OneHotCategorical(
            logits=latent_dim_logit, dtype=tf.float32).sample()
        assignments = tf.concat([assignments, sample], axis=1)
      assignments = assignments[:, 1:, :]
    else:
      logits = tf.zeros([10, FLAGS.latent_size, FLAGS.num_codes])
      assignments = tf.reduce_mean(tfd.OneHotCategorical(
          logits=logits, dtype=tf.float32).sample(1), axis=0)

    prior_samples = tf.reduce_sum(
        assignments[..., tf.newaxis] *
        tf.reshape(vector_quantizer.codebook,
                   [1, 1, FLAGS.num_codes, FLAGS.code_size]),
        axis=2)
    prior_samples = prior_samples[tf.newaxis, ...]
    decoded_distribution_given_random_prior = decoder(prior_samples)
    random_images = decoded_distribution_given_random_prior.mean()[0]

    # Save summaries.
    tf.summary.image("train_inputs",
                     tf.cast(images, tf.float32),
                     max_outputs=10,
                     collections=["train_image"])
    tf.summary.image("train_reconstructions",
                     reconstructed_images,
                     max_outputs=10,
                     collections=["train_image"])
    tf.summary.image("train_prior_samples",
                     tf.cast(random_images, tf.float32),
                     max_outputs=10,
                     collections=["train_image"])
    tf.summary.image("heldout_inputs",
                     tf.cast(images, tf.float32),
                     max_outputs=10,
                     collections=["heldout_image"])
    tf.summary.image("heldout_reconstructions",
                     reconstructed_images,
                     max_outputs=10,
                     collections=["heldout_image"])
    tf.summary.scalar("heldout/marginal_loss",
                      marginal_nll,
                      collections=["heldout"])

    # Perform inference by minimizing the loss function.
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if FLAGS.num_iaf_flows > 0:
      encoder_variables = encoder.variables
      iaf_variables = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope="iaf_variables")
      grads_and_vars = optimizer.compute_gradients(loss)
      grads_and_vars_except_encoder = [
          x for x in grads_and_vars if x[1] not in encoder_variables]
      grads_and_vars_except_iaf = [
          x for x in grads_and_vars if x[1] not in iaf_variables]

      def train_op_except_iaf():
        return optimizer.apply_gradients(
            grads_and_vars_except_iaf,
            global_step=global_step)

      def train_op_except_encoder():
        return optimizer.apply_gradients(
            grads_and_vars_except_encoder,
            global_step=global_step)

      def train_op_all():
        return optimizer.apply_gradients(
            grads_and_vars,
            global_step=global_step)

      if FLAGS.stop_training_encoder_after_startup:
        after_startup_train_op = train_op_except_encoder
      else:
        after_startup_train_op = train_op_all

      train_op = tf.cond(
          global_step < FLAGS.iaf_startup_steps,
          true_fn=train_op_except_iaf,
          false_fn=after_startup_train_op)
    else:
      train_op = optimizer.minimize(loss, global_step=global_step)

    summary = tf.summary.merge_all()
    heldout_summary = tf.summary.merge_all(key="heldout")
    train_image_summary = tf.summary.merge_all(key="train_image")
    heldout_image_summary = tf.summary.merge_all(key="heldout_image")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session(FLAGS.master) as sess:
      summary_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
      sess.run(init)

      # Run the training loop.
      train_handle = sess.run(training_iterator.string_handle())
      heldout_handle = sess.run(heldout_iterator.string_handle())
      for step in range(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict={handle: train_handle})
        duration = time.time() - start_time
        if step % 100 == 0:
          marginal_nll_val = sess.run(marginal_nll,
                                      feed_dict={handle: heldout_handle})
          print("Step: {:>3d} Training Loss: {:.3f} Heldout NLL: {:.3f} "
                "({:.3f} sec)".format(step, loss_value, marginal_nll_val,
                                      duration))

          # Update the events file.
          summary_str = sess.run(summary, feed_dict={handle: train_handle})
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

          summary_str_heldout = sess.run(heldout_summary,
                                         feed_dict={handle: heldout_handle})
          summary_writer.add_summary(summary_str_heldout, step)
          summary_writer.flush()

        # Periodically save a checkpoint and visualize model progress.
        if (step + 1) % FLAGS.viz_steps == 0:
          summary_str_train_images = sess.run(
              train_image_summary,
              feed_dict={handle: train_handle})
          summary_str_heldout_images = sess.run(
              heldout_image_summary,
              feed_dict={handle: heldout_handle})
          summary_writer.add_summary(summary_str_train_images, step)
          summary_writer.add_summary(summary_str_heldout_images, step)
          checkpoint_file = os.path.join(FLAGS.model_dir, "model.ckpt")
          saver.save(sess, checkpoint_file, global_step=step)


if __name__ == "__main__":
  tf.app.run()
