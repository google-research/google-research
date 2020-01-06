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

"""Bottom and top transformations of the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers

import tensorflow.compat.v1 as tf
import state_of_sparsity.layers.l0_regularization as l0
import state_of_sparsity.layers.variational_dropout as vd
from state_of_sparsity.sparse_transformer.layers import common_sparse
from tensorflow.contrib.eager.python import tfe as contrib_eager
from tensorflow.contrib.model_pruning.python import pruning


# TODO(tgale): This is a hack. Find a better way to avoid collecting
# duplicate weight variables for variation dropout and l0-regularization
COLLECTED_VARIABLES = False


def _get_weights(model_hparams, vocab_size, hidden_dim=None):
  """Create or get concatenated embedding or softmax variable.

  Args:
    model_hparams: tf.HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.
    hidden_dim: dim of the variable. Defaults to model_hparams.hidden_size

  Returns:
     a list of num_shards Tensors.
  """
  if hidden_dim is None:
    hidden_dim = model_hparams.hidden_size
  num_shards = model_hparams.symbol_modality_num_shards
  shards = []

  sparsity_technique = model_hparams.get("sparsity_technique")
  aux_params_shards = []
  for i in range(num_shards):
    shard_size = (vocab_size // num_shards) + (
        1 if i < vocab_size % num_shards else 0)
    var_name = "weights_%d" % i

    weight_init_stddev = hidden_dim**-0.5
    if (model_hparams.get("load_masks_from") and
        model_hparams.get("initial_sparsity")):
      # If we are loading constant masks for scratch-e or scratch-b
      # experiments, we optionally rescale the variance of the weight
      # initialization.
      initial_sparsity = model_hparams.get("initial_sparsity")
      weight_init_stddev = (hidden_dim * (1 - initial_sparsity))**-0.5
      tf.logging.info("Using sparse initialization with sparsity {} for symbol "
                      .format(initial_sparsity))

    shards.append(
        tf.get_variable(
            var_name, [shard_size, hidden_dim],
            initializer=tf.random_normal_initializer(0.0, weight_init_stddev)))
    if sparsity_technique == "variational_dropout":
      aux_params_shards.append(
          tf.get_variable(
              var_name + "_aux", [shard_size, hidden_dim],
              initializer=tf.constant_initializer(value=-10.0)))
    elif sparsity_technique == "l0_regularization":
      initializer = tf.random_normal_initializer(mean=2.197, stddev=0.01)
      aux_params_shards.append(
          tf.get_variable(
              var_name + "_aux", [shard_size, hidden_dim],
              initializer=initializer))

  if num_shards == 1:
    ret = shards[0]
  else:
    ret = tf.concat(shards, 0)

  if not aux_params_shards:
    # Convert ret to tensor.
    if not contrib_eager.in_eager_mode():
      ret = common_layers.convert_gradient_to_tensor(ret)
    return ret

  # Handle the auxiliary parameters
  if num_shards == 1:
    aux_ret = aux_params_shards[0]
  else:
    aux_ret = tf.concat(aux_params_shards, 0)

  global COLLECTED_VARIABLES
  if not COLLECTED_VARIABLES:
    if sparsity_technique == "variational_dropout":
      tf.add_to_collection(
          common_sparse.VARIATIONAL_DROPOUT_PARAMETERS,
          (ret, aux_ret))
    elif sparsity_technique == "l0_regularization":
      tf.add_to_collection(
          common_sparse.L0_REGULARIZATION_PARAMETERS,
          (ret, aux_ret))
    COLLECTED_VARIABLES = True

  # Convert aux ret to tensor.
  if not contrib_eager.in_eager_mode():
    ret = common_layers.convert_gradient_to_tensor(ret)
    aux_ret = common_layers.convert_gradient_to_tensor(aux_ret)
  return (ret, aux_ret)


def bottom_simple(x, model_hparams, vocab_size, name, reuse):
  """Bottom transformation."""
  with tf.variable_scope(name, reuse=reuse):
    # Ensure the inputs are 3-D
    if len(x.get_shape()) == 4:
      x = tf.squeeze(x, axis=3)
    while len(x.get_shape()) < 3:
      x = tf.expand_dims(x, axis=-1)

    var = _get_weights(model_hparams, vocab_size)
    x = common_layers.dropout_no_scaling(
        x, 1.0 - model_hparams.symbol_dropout)

    sparsity_technique = model_hparams.get("sparsity_technique")
    training = model_hparams.get("mode") == tf.estimator.ModeKeys.TRAIN
    if sparsity_technique == "variational_dropout":
      if training:
        ret = vd.nn.embedding_lookup_train(
            var,
            x,
            clip_alpha=model_hparams.get("clip_log_alpha"))
      else:
        threshold = model_hparams.get("log_alpha_threshold")
        ret = vd.nn.embedding_lookup_eval(
            var,
            x,
            threshold=threshold)
    elif sparsity_technique == "l0_regularization":
      if training:
        ret = l0.nn.embedding_lookup_train(var, x)
      else:
        ret = l0.nn.embedding_lookup_eval(var, x)
    elif (sparsity_technique == "magnitude_pruning" or
          sparsity_technique == "random_pruning"):
      ret = common_layers.gather(pruning.apply_mask(var), x)
    else:
      ret = common_layers.gather(var, x)

    # post-process the embedding vectors
    if model_hparams.multiply_embedding_mode == "sqrt_depth":
      ret *= model_hparams.hidden_size**0.5
    ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
    return ret


def bottom(x, model_hparams, vocab_size):
  """Bottom transformation for symbols."""
  # Sparsity techniques only support shared weight matrices for now
  sparsity_technique = model_hparams.get("sparsity_technique")
  assert (not sparsity_technique or
          model_hparams.shared_embedding_and_softmax_weights)

  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    return bottom_simple(
        x, model_hparams, vocab_size, "shared", reuse=None)
  return bottom_simple(
      x, model_hparams, vocab_size, "input_emb", reuse=None)


def targets_bottom(x, model_hparams, vocab_size):
  """Bottom transformation for target symbols."""
  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    try:
      return bottom_simple(
          x, model_hparams, vocab_size, "shared", reuse=True)
    except ValueError:
      # perhaps there were no inputs, and this is a new variable.
      return bottom_simple(
          x, model_hparams, vocab_size, "shared", reuse=None)
  else:
    return bottom_simple(
        x, model_hparams, vocab_size, "target_emb", reuse=None)


def top(body_output, targets, model_hparams, vocab_size):
  """Generate logits.

  Args:
    body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
    targets: Unused.
    model_hparams: tf.HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
  """
  del targets  # unused arg
  # Sparsity techniques only support shared weight matrices for now
  sparsity_technique = model_hparams.get("sparsity_technique")
  assert (not sparsity_technique or
          model_hparams.shared_embedding_and_softmax_weights)
  if model_hparams.shared_embedding_and_softmax_weights:
    scope_name = "shared"
    reuse = tf.AUTO_REUSE
  else:
    scope_name = "softmax"
    reuse = False

  with tf.variable_scope(scope_name, reuse=reuse):
    body_output_shape = common_layers.shape_list(body_output)
    var = _get_weights(model_hparams, vocab_size, body_output_shape[-1])
    if (model_hparams.factored_logits and
        model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # Sparsity techniques only support non-factored logits for now
      assert not sparsity_technique

      # insert channels dimension
      body_output = tf.expand_dims(body_output, 3)
      return common_layers.FactoredTensor(body_output, var)
    else:
      body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])

      training = model_hparams.get("mode") == tf.estimator.ModeKeys.TRAIN
      if sparsity_technique == "variational_dropout":
        if training:
          logits = vd.nn.matmul_train(
              body_output,
              var,
              transpose_b=True,
              clip_alpha=model_hparams.get("clip_log_alpha"))
        else:
          threshold = model_hparams.get("log_alpha_threshold")
          logits = vd.nn.matmul_eval(
              body_output,
              var,
              transpose_b=True,
              threshold=threshold)
      elif sparsity_technique == "l0_regularization":
        if training:
          logits = l0.nn.matmul_train(
              body_output,
              var,
              transpose_b=True)
        else:
          logits = l0.nn.matmul_eval(
              body_output,
              var,
              transpose_b=True)
      elif (sparsity_technique == "magnitude_pruning" or
            sparsity_technique == "random_pruning"):
        logits = tf.matmul(
            body_output,
            pruning.apply_mask(var),
            transpose_b=True)
      else:
        logits = tf.matmul(body_output, var, transpose_b=True)

      return tf.reshape(
          logits,
          body_output_shape[:-1] + [1, vocab_size])
