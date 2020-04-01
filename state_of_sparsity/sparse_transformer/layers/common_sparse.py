# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Sparse versions of common layers.

TODO(tgale): The hparams passed to the dense layer should stored in an
hparams object so that the API is not specific to one technique. The
different regularizer weight curves should maybe be separated out into a
different file, along with the loss terms themselves. The dense layer can be
made much much more clean.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import activations
from tensorflow.compat.v1.keras import initializers

import state_of_sparsity.layers.l0_regularization as l0
import state_of_sparsity.layers.variational_dropout as vd
from state_of_sparsity.sparse_transformer.layers import common_init
from tensorflow.contrib.model_pruning.python.layers import layers as pruning_layers


L0_REGULARIZATION_PARAMETERS = "theta_and_log_alpha"
VARIATIONAL_DROPOUT_PARAMETERS = "theta_and_log_sigma2"


def dense(
    x,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    sparsity_technique="variational_dropout",
    auxiliary_initializer=None,
    threshold=3.0,
    clip_alpha=None,
    training=True,
    dtype=tf.float32,
    name=None,
    initial_sparsity=None):
  """Matmul & bias add that supports broadcasting for batched gemm.

  Supports a contrained set of functionality provided by tf.layers.dense.

  Args:
    x: input tensor.
    units: number of units in the dense layer.
    activation: activation function to use in the layer.
    use_bias: whether or not to add a bias to the output.
    kernel_initializer: weight initializer for the layer.
    bias_initializer: weight initializer for the bias.
    sparsity_technique: sparsification technique to apply to the weights.
    auxiliary_initializer: initializer for auxiliary variables use in
      variational dropout and l0 regularization.
    threshold: log-alpha threshold for variational dropout.
    clip_alpha: whether to clip the alpha values for variational dropout.
    training: whether this run is training or evaluation the model.
    dtype: data type for the weights and computation.
    name: name for the layer.
    initial_sparsity: initial weight sparsity at the start of training.

  Returns:
    Tensor representing the output of the layer.
  """
  activation = activations.get(activation)
  kernel_initializer = initializers.get(kernel_initializer)
  bias_initializer = initializers.get(bias_initializer)

  if (sparsity_technique == "magnitude_pruning" or
      sparsity_technique == "random_pruning"):
    if initial_sparsity is not None:
      # If the initial sparsity value is passed in, use the sparse glorot
      # uniform initializer to account for the zero valued weights.
      kernel_initializer = common_init.SparseGlorotUniform(
          initial_sparsity, dtype=dtype)
      tf.logging.info(
          "Using sparse initialization with sparsity {} for variable {}"
          .format(initial_sparsity, tf.get_variable_scope().name))

    # If the sparsity technique is magnitude_pruning, or random_pruning
    # use the model_pruning masked_fully_connected layer
    #
    # masked_fully_connected doesn't take use_bias arg, pass None for the
    # bias initializer if we don't want a bias variable
    bias_initializer = bias_initializer if use_bias else None
    with tf.variable_scope(name, default_name="dense"):
      return pruning_layers.masked_fully_connected(
          inputs=x,
          num_outputs=units,
          activation_fn=activation,
          weights_initializer=kernel_initializer,
          biases_initializer=bias_initializer)
  if initial_sparsity is not None:
    raise ValueError("initial_sparsity only supported for mp & rp")

  # layer_name = "%s_{}" % name if name else "{}"

  input_shape = x.get_shape().as_list()
  if input_shape[-1] is None:
    raise ValueError("The last dimension of the inputs to `Dense` "
                     "should be defined. Found `None`.")

  with tf.variable_scope(name, default_name="dense") as vs:
    kernel = tf.get_variable(
        "kernel",
        shape=[input_shape[-1], units],
        initializer=kernel_initializer,
        dtype=dtype,
        trainable=True)

    bias = None
    if use_bias:
      bias = tf.get_variable(
          "bias",
          shape=[units,],
          initializer=bias_initializer,
          dtype=dtype,
          trainable=True)

  # Compute the dense layer
  if sparsity_technique == "variational_dropout":
    log_sigma2_initializer = initializers.get(auxiliary_initializer)

    if not log_sigma2_initializer:
      log_sigma2_initializer = tf.constant_initializer(value=-10, dtype=dtype)

    with tf.variable_scope(vs, auxiliary_name_scope=False) as vs1:
      with tf.name_scope(vs1.original_name_scope):
        log_sigma2 = tf.get_variable(
            "log_sigma2",
            shape=[input_shape[-1], units],
            initializer=log_sigma2_initializer,
            dtype=dtype,
            trainable=True)

    variational_parameters = (kernel, log_sigma2)
    tf.add_to_collection(
        VARIATIONAL_DROPOUT_PARAMETERS,
        variational_parameters)

    input_rank = x.get_shape().ndims
    if input_rank > 2:
      if training:
        outputs = vd.nn.broadcast_matmul_train(
            x,
            variational_parameters,
            clip_alpha=clip_alpha)
      else:
        outputs = vd.nn.broadcast_matmul_eval(
            x,
            variational_parameters,
            threshold)
    else:
      if training:
        outputs = vd.nn.matmul_train(
            x,
            variational_parameters,
            clip_alpha=clip_alpha)
      else:
        outputs = vd.nn.matmul_eval(
            x,
            variational_parameters,
            threshold)
  else:
    if sparsity_technique != "l0_regularization":
      raise ValueError("Unsupported sparsity technique {}"
                       .format(sparsity_technique))
    log_alpha_initializer = initializers.get(auxiliary_initializer)

    if not log_alpha_initializer:
      # Default to \alpha / (\alpha + 1) equal to 0.5
      # Default to \alpha / (\alpha + 1) = .1
      log_alpha_initializer = tf.random_normal_initializer(
          mean=2.197, stddev=0.01, dtype=dtype)

    with tf.variable_scope(vs, auxiliary_name_scope=False) as vs1:
      with tf.name_scope(vs1.original_name_scope):
        log_alpha = tf.get_variable(
            "log_alpha",
            shape=[input_shape[-1], units],
            initializer=log_alpha_initializer,
            dtype=dtype,
            trainable=True)

    weight_parameters = (kernel, log_alpha)
    tf.add_to_collection(
        L0_REGULARIZATION_PARAMETERS,
        weight_parameters)

    input_rank = x.get_shape().ndims
    if input_rank > 2:
      if training:
        outputs = l0.nn.broadcast_matmul_train(x, weight_parameters)
      else:
        outputs = l0.nn.broadcast_matmul_eval(x, weight_parameters)
    else:
      if training:
        outputs = l0.nn.matmul_train(x, weight_parameters)
      else:
        outputs = l0.nn.matmul_eval(x, weight_parameters)

  # Handle the bias and activation
  if use_bias:
    outputs = tf.nn.bias_add(outputs, bias)
  if activation is not None:
    return activation(outputs)
  return outputs


class CubicDKLWeight(object):
  """Helper class for cubic dkl weight schedule."""

  def __init__(self, dkl_weight, begin_step, end_step, weight_decay=0.9):
    assert end_step > begin_step
    self.dkl_weight = dkl_weight
    self.begin_step = tf.constant(begin_step, dtype=tf.float32)
    self.end_step = tf.constant(end_step, dtype=tf.float32)
    self.weight_decay = tf.constant(weight_decay, dtype=tf.float32)
    self.num_steps = self.end_step - self.begin_step

    self.current_weight = tf.get_variable(
        "dkl_weight",
        shape=[],
        initializer=tf.zeros_initializer(),
        trainable=False)

  def _exp_moving_average(self, iteration):
    weight = self._cubic_weight(iteration)

    new_weight = tf.add(
        tf.multiply(weight, 1 - self.weight_decay),
        tf.multiply(self.current_weight, self.weight_decay))

    updated_weight = tf.assign(self.current_weight, new_weight)
    return updated_weight

  def _cubic_weight(self, iteration):
    exp_base = (tf.maximum(tf.cast(iteration, tf.float32)
                           - self.begin_step, 0.0)
                / self.num_steps)
    return self.dkl_weight * tf.minimum(1.0 - tf.pow(1.0 - exp_base, 3), 1.0)

  def __call__(self, iteration):
    return self._exp_moving_average(iteration)


class LinearDKLWeight(object):
  """Helper class for linear dkl weight schedule."""

  def __init__(self, dkl_weight, begin_step, end_step):
    assert end_step > begin_step
    self.dkl_weight = dkl_weight
    self.begin_step = begin_step
    self.end_step = end_step

  def __call__(self, i):
    current_weight = tf.maximum(
        0.0,
        tf.cast(i, tf.float32) - self.begin_step)
    current_weight = current_weight / (self.end_step - self.begin_step)
    current_weight = tf.minimum(current_weight, 1.0)
    return current_weight * self.dkl_weight


def variational_dropout_dkl_loss(
    sparsity_check=True,
    threshold=3.0,
    weight_function="linear",
    dkl_weight=1.0,
    begin_step=0.,
    end_step=0.,
    clip_alpha=None):
  """Computes the KL divergance loss term for all parameters."""
  variational_parameters = tf.get_collection(VARIATIONAL_DROPOUT_PARAMETERS)

  if sparsity_check:
    check_weight_sparsity(
        variational_parameters,
        technique="variational_dropout",
        threshold=threshold)

  if weight_function == "linear":
    dkl_weight_fn = LinearDKLWeight(dkl_weight, begin_step, end_step)
  elif weight_function == "cubic":
    dkl_weight_fn = CubicDKLWeight(dkl_weight, begin_step, end_step)
  else:
    raise ValueError("Invalid weight function {}".format(weight_function))

  # Calculate the kl-divergence weight for this iteration
  iteration = tf.train.get_or_create_global_step()
  current_weight = dkl_weight_fn(iteration)

  # Compute the dkl over the parameters and weight it
  dkl_loss = [
      vd.nn.negative_dkl(p, clip_alpha=clip_alpha)
      for p in variational_parameters
  ]
  dkl_loss = current_weight * tf.add_n(dkl_loss)

  # Add summary for the kl-divergence weight and weighted loss
  tf.summary.scalar("dkl_weight", current_weight)
  tf.summary.scalar("weighted_dkl_loss", dkl_loss)

  return [dkl_loss]


def l0_regularization_term(
    sparsity_check=True,
    regularization_weight=1.0,
    weight_start=0.0,
    weight_end=0.0,
    weight_function="linear"):
  """Computes the expected l0-norm over all the parameters.

  Args:
    sparsity_check: whether to add summaries for the global weight sparsity.
    regularization_weight: base weight for the total l0 norm.
    weight_start: iteration to start the regularizer ramp-up.
    weight_end: iteration to end the regularization ramp-up.
    weight_function: either "linear" or "cubic".

  Returns:
    Single item list with Tensor of weighted l0-norm loss contribution.
  """
  weight_parameters = tf.get_collection(L0_REGULARIZATION_PARAMETERS)

  if sparsity_check:
    check_weight_sparsity(
        weight_parameters,
        technique="l0_regularization")

  if weight_function == "linear":
    dkl_weight_fn = LinearDKLWeight(
        regularization_weight,
        weight_start,
        weight_end)
  elif weight_function == "cubic":
    dkl_weight_fn = CubicDKLWeight(
        regularization_weight,
        weight_start,
        weight_end)
  else:
    raise ValueError("Invalid weight function {}".format(weight_function))

  # Calculate the l0-regularization weight for this iteration
  iteration = tf.train.get_or_create_global_step()
  current_weight = dkl_weight_fn(iteration)

  l0_regularization = [l0.nn.l0_norm(a) for (_, a) in weight_parameters]
  l0_regularization = current_weight * tf.add_n(l0_regularization)

  # Add summary for the kl-divergence weight and weighted loss
  tf.summary.scalar("l0_norm_weight", current_weight)
  tf.summary.scalar("weighted_l0_norm", l0_regularization)
  return [l0_regularization]


def check_weight_sparsity(
    weights,
    technique="variational_dropout",
    threshold=3.0,
    dtype=tf.float32):
  """Helper function for calculating global weight sparsity.

  Args:
    weights: List of weight tensors.
    technique: sparsity technique.
    threshold: log alpha threshold for variational dropout.
    dtype: datatype for computation.

  Returns:
    Tensors of the calculated global weight sparsity and sparsity per-layer.
  """
  zero_per_layer = []
  weights_per_layer = []
  sparsity_per_layer = []
  with tf.name_scope("weight_sparsity"):
    for i in range(len(weights)):
      w = weights[i]

      if isinstance(w, tuple):
        assert len(w) == 2
        if technique == "variational_dropout":
          theta, log_sigma2 = w

          # Compute the weight mask based on the set threshold
          log_alpha = vd.common.compute_log_alpha(
              log_sigma2, theta, value_limit=None)
          w = tf.cast(tf.less(log_alpha, threshold), dtype)
        else:
          assert technique == "l0_regularization"
          theta, log_alpha = w

          # Compute the evaluation time weights
          weight_noise = l0.common.hard_concrete_mean(log_alpha)
          w = theta * weight_noise

      # NOTE: This is much more complex than just calling tf.nn.zero_fraction
      # but we get the number of zero weights in each matrix which lets us
      # calculate global sparsity easily
      is_zero = tf.cast(tf.equal(w, tf.constant(0, dtype=tf.float32)), dtype)
      zero = tf.reduce_sum(is_zero)

      total_weights = tf.cast(tf.size(w), dtype)
      sparsity = zero / total_weights

      sparsity_per_layer.append(sparsity)
      zero_per_layer.append(zero)
      weights_per_layer.append(total_weights)

    # Calculate the global sparsity percentage
    total_zero = tf.add_n(zero_per_layer)
    total_weights = tf.add_n(weights_per_layer)
    sparsity = (total_zero / total_weights)

    tf.summary.scalar("global_weight_sparsity", sparsity)
  return sparsity, sparsity_per_layer
