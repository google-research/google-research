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

"""Sparse Image Transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow.compat.v1 as tf

from routing_transformer import utils

_IMGS = {}


@registry.register_model
class SparseImagetransformer(t2t_model.T2TModel):
  """Sparse Image Transformer."""

  @property
  def inputs_vocab_size(self):
    """Size of vocabulary for 'inputs' feature."""
    return self.problem_hparams.vocab_size["inputs"]

  @property
  def targets_vocab_size(self):
    """Size of vocabulary for 'targets' feature."""
    return self.problem_hparams.vocab_size["targets"]

  @property
  def frame_height(self):
    return self.hparams.frame_height

  @property
  def frame_width(self):
    return self.hparams.frame_width

  @property
  def num_channels(self):
    return self.hparams.problem.num_channels

  @property
  def batch_size(self):
    return self.hparams.batch_size

  @property
  def hidden_size(self):
    return self.hparams.hidden_size

  @property
  def num_decoder_layers(self):
    return self.hparams.num_decoder_layers

  @property
  def add_positional_emb(self):
    return self.hparams.add_positional_emb

  @property
  def is_inputs_class_label(self):
    """True if 'inputs' feature represents a class label."""
    return (self.problem_hparams.modality["inputs"] ==
            modalities.ModalityType.CLASS_LABEL)

  @property
  def is_decode(self):
    return self.hparams.mode == tf.estimator.ModeKeys.PREDICT

  def get_shape_for_decoder(self):
    """Returns the shape of the sequence to be fed into the decoder."""
    if len(self.hparams.query_shape) == 2:
      return [self.frame_height, self.frame_width * self.num_channels]
    elif len(self.hparams.query_shape) == 1:
      return [self.frame_height * self.frame_width * self.num_channels]
    else:
      raise ValueError("Only local 1D and local 2D attention is supported.")

  def process_partial_targets_decoding(self, targets):
    """Processes partially generated targets in decoding mode."""
    original_shape = self.get_shape_for_decoder()
    blocks_per_dim = [
        s // q for s, q in zip(original_shape, self.hparams.query_shape)
    ]
    targets_shape = utils.shape_list(targets)
    targets = tf.reshape(
        targets, [targets_shape[0], -1,
                  np.prod(self.hparams.query_shape), 1])

    targets = utils.unflatten_blocks_nd(targets, blocks_per_dim)
    targets = utils.put_back_blocks_nd(targets, self.hparams.query_shape)
    targets = tf.reshape(
        targets, [-1, self.frame_height, self.frame_width, self.num_channels])
    return targets

  def prepare_decoder(self, targets):
    """Prepares targets for transformer decoder."""
    shape = utils.shape_list(targets)
    # image should be [batch, height, width, channels]
    assert len(shape) == 4, "Image tensors should be 4-dimensional"

    # Shift positions
    targets = tf.reshape(targets, [-1] + self.get_shape_for_decoder() + [1])
    targets = utils.right_shift_blockwise_nd(targets, self.hparams.query_shape)

    # Add channel embeddings
    targets = tf.reshape(
        targets, [-1, self.frame_height, self.frame_width, self.num_channels])
    targets = utils.get_channel_embeddings(
        io_depth=self.num_channels,
        targets=targets,
        hidden_size=self.hidden_size)

    # add positional embeddings if needed
    if self.add_positional_emb:
      targets = utils.add_positional_embedding_nd(
          targets,
          max_length=max(self.frame_height, self.frame_width,
                         self.num_channels),
          name="pos_emb")
    targets = tf.reshape(targets, [-1] + self.get_shape_for_decoder() +
                         [self.hidden_size])
    return targets

  def multinomial_squeeze(self, logits, temperature=1.0):
    """multinomial sampling from logits."""
    logits_shape = utils.shape_list(logits)
    reshaped_logits = (tf.reshape(logits, [-1, logits_shape[-1]]) / temperature)
    choices = tf.multinomial(reshaped_logits, 1)
    choices = tf.reshape(choices, logits_shape[:-1])
    return tf.to_int32(choices)

  def produce_output(self, decoder_output):
    """Maps decoder output to final logits."""
    # map decoder output to output vocab size
    output = tf.layers.dense(
        decoder_output,
        self.targets_vocab_size,
        activation=None,
        name="final_dense")

    if self.is_decode:
      return output

    # Reshape to a normal image
    output = tf.reshape(output, [
        -1, self.frame_height, self.frame_width, self.num_channels,
        self.targets_vocab_size
    ])
    return output

  def body(self, features, decode_step=None, cache=None, decoding_stats=None):
    targets = tf.to_int32(features["targets"])
    if self.is_decode:
      targets = self.process_partial_targets_decoding(targets)
    decoder_input = self.prepare_decoder(targets)
    extra_losses = []

    if not self.hparams.unconditional:  # condition on class label
      if not self.is_inputs_class_label:
        raise ValueError("SparseImagetransformer can only condition on "
                         "'inputs' feature if it represents class label.")
      inputs = features["inputs"]

      # Embed class here rather than in bottom().
      if inputs.dtype not in [tf.int32, tf.int64]:
        raise ValueError("Do not embed 'inputs' before body(). "
                         "Found dtype=%s." % inputs.dtype)
      inputs = utils.get_embeddings(
          targets=inputs,
          vocab_size=self.inputs_vocab_size,
          hidden_size=self.hidden_size,
          name="class_conditional_embedding")

      # Add class embedding to each spatial location.
      batch_size = tf.shape(targets)[0]
      hidden_size = tf.shape(inputs)[-1]
      num_middle_dims = len(decoder_input.shape) - 2
      decoder_input += tf.reshape(inputs, [batch_size] + [1] * num_middle_dims +
                                  [hidden_size])

    decoder_output = utils.transformer_decoder_layers(
        inputs=decoder_input,
        num_layers=self.num_decoder_layers,
        hparams=self.hparams,
        decode_step=decode_step,
        losses=extra_losses,
        cache=cache,
        name="decoder",
        decoding_stats=decoding_stats)
    logits = self.produce_output(decoder_output)

    # Return logits as-is in decoding mode
    if self.is_decode:
      return logits

    # Produce a summary of the output.
    results = self.multinomial_squeeze(logits, self.hparams.sampling_temp)
    results = tf.reshape(
        results, [-1, self.frame_height, self.frame_width, self.num_channels])
    if utils.is_xla_compiled():
      _IMGS["predictions"] = results

    # Prepare loss.
    loss_dict = {}
    if extra_losses:
      loss_dict["extra_loss"] = tf.add_n(extra_losses)
    return logits, loss_dict

  def top(self, body_outputs, features):
    return body_outputs

  def loss(self, logits, features):
    # Add cross entropy loss
    targets = features["targets"]
    one_hot_targets = tf.one_hot(
        tf.cast(targets, dtype=tf.int32), self.targets_vocab_size)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_targets, logits=logits)
    return loss

  def sample(self, features, decode_step, cache, decoding_stats):
    """Sample step for infer."""
    with tf.variable_scope("sparse_imagetransformer/body", reuse=tf.AUTO_REUSE):
      logits = self.body(features, decode_step, cache, decoding_stats)
      logits = tf.reshape(logits, [self.batch_size, self.targets_vocab_size])
      sample = self.multinomial_squeeze(logits, self.hparams.sampling_temp)
      sample = tf.reshape(sample, [self.batch_size])
      return sample, logits

  def infer(self, features, **kwargs):
    decode_length = (self.frame_height * self.frame_width * self.num_channels)
    cache = {}
    decoding_stats = {}
    targets_old = features.get("targets", None)
    initial_output = tf.zeros((self.batch_size, decode_length), dtype=tf.int32)
    initial_logits = tf.zeros(
        (self.batch_size, decode_length, self.targets_vocab_size))
    # call body once to initialize cache with representations of input frames.
    features["targets"] = initial_output
    with tf.variable_scope(
        "sparse_imagetransformer/body", reuse=tf.AUTO_REUSE, use_resource=True):
      self.body(
          features,
          decode_step=None,
          cache=cache,
          decoding_stats=decoding_stats)

    def infer_step(i, recent_output, recent_logits, cache, decoding_stats):
      """Inference step."""
      features_copy = features.copy()
      features_copy["targets"] = recent_output
      cur_sample, cur_logit = self.sample(
          features_copy,
          decode_step=i,
          cache=cache,
          decoding_stats=decoding_stats)
      pos = i
      samples = recent_output + tf.scatter_nd(
          indices=[[b, pos] for b in range(self.batch_size)],
          updates=cur_sample,
          shape=utils.shape_list(recent_output))
      logits = recent_logits + tf.scatter_nd(
          indices=[[b, pos] for b in range(self.batch_size)],
          updates=cur_logit,
          shape=utils.shape_list(recent_logits))
      return i + 1, samples, logits, cache, decoding_stats

    def while_exit_cond(i, result, logits, cache, decoding_stats):  # pylint: disable=unused-argument
      """Exit the loop if it reaches decode_length."""
      not_overflow = i < decode_length
      return not_overflow

    _, final_result, final_logits, _, decoding_stats = tf.while_loop(
        while_exit_cond,
        infer_step,
        [tf.constant(0), initial_output, initial_logits, cache, decoding_stats],
        back_prop=False,
        parallel_iterations=1)

    original_shape = self.get_shape_for_decoder()

    blocks_per_dim = [
        s // q for s, q in zip(original_shape, self.hparams.query_shape)
    ]
    final_result_shape = utils.shape_list(final_result)
    final_result = tf.reshape(
        final_result,
        [final_result_shape[0], -1,
         np.prod(self.hparams.query_shape), 1])
    final_logits_shape = utils.shape_list(final_logits)
    final_logits = tf.reshape(final_logits, [
        final_logits_shape[0], -1,
        np.prod(self.hparams.query_shape), final_logits_shape[-1]
    ])
    final_result = utils.unflatten_blocks_nd(final_result, blocks_per_dim)
    final_result = utils.put_back_blocks_nd(final_result,
                                            self.hparams.query_shape)
    final_logits = utils.unflatten_blocks_nd(final_logits, blocks_per_dim)
    final_logits = utils.put_back_blocks_nd(final_logits,
                                            self.hparams.query_shape)

    final_result = tf.reshape(
        final_result,
        [-1, self.frame_height, self.frame_width, self.num_channels])
    final_logits = tf.reshape(final_logits, [
        -1, self.frame_height, self.frame_width, self.num_channels,
        self.targets_vocab_size
    ])

    if utils.is_xla_compiled():
      _IMGS["decodes"] = final_result

    for name, value in decoding_stats.items():
      tf.summary.scalar("decodes/%s" % name, value / decode_length)

    # Reassign targets back to the previous value.
    if targets_old is not None:
      features["targets"] = targets_old

    return {
        "outputs": final_result,
        "scores": None,
        "logits": final_logits,
        "losses": None,
    }


def _targets_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # Unused
  if utils.is_xla_compiled():
    _IMGS["targets"] = x
  else:
    tf.summary.image("targets", tf.cast(x, dtype=tf.uint8))
  return x


def _inputs_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # Unused
  return x


@registry.register_hparams
def sparse_imagetransformer_base():
  """Sparse image transformer base hparams with local 1D attention."""
  hparams = common_hparams.basic_params1()

  # Basic HParams
  hparams.hidden_size = 256
  hparams.batch_size = 1  # Per TPU core
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer = "adafactor"
  hparams.optimizer_adafactor_beta1 = 0.0
  hparams.optimizer_adafactor_beta2 = 0.999
  hparams.optimizer_adafactor_clipping_threshold = 1.0
  hparams.optimizer_adafactor_decay_type = "pow"
  hparams.optimizer_adafactor_memory_exponent = 0.8
  hparams.optimizer_adafactor_multiply_by_parameter_scale = True
  hparams.learning_rate_schedule = "constant*rsqrt_normalized_decay"
  hparams.learning_rate_warmup_steps = 10000
  hparams.learning_rate_constant = 0.01
  hparams.initializer_gain = 0.2
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.label_smoothing = 0.0
  hparams.bottom = {
      # Bypass bottom().
      "inputs": _inputs_bottom,
      "targets": _targets_bottom,
  }
  hparams.tpu_enable_host_call = True  # Enable summaries on TPU
  hparams.add_hparam("add_positional_emb", True)
  hparams.add_hparam("frame_height", 32)
  hparams.add_hparam("frame_width", 32)

  # Transformer HParams
  hparams.norm_type = "layer"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.add_hparam("num_decoder_layers", 12)
  hparams.add_hparam("local_num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)  # Uses hidden_size
  hparams.add_hparam("attention_value_channels", 0)  # Uses hidden_size
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")
  hparams.add_hparam("filter_size", 256)  # Used in ffn_layer
  hparams.add_hparam("relu_dropout", 0.0)  # Used in ffn_layer

  # Local 1D HParams
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)
  hparams.add_hparam("query_shape", (128,))
  hparams.add_hparam("memory_query_shape", (128,))
  hparams.add_hparam("memory_flange", (128,))

  # Sparsity HParams
  hparams.add_hparam("sparsity_cluster_size", 0)
  hparams.add_hparam("sparsity_cluster_attention_window", 0)
  hparams.add_hparam("sparsity_cluster_num_heads", 0)
  hparams.add_hparam("sparsity_strided_num_heads", 0)
  hparams.add_hparam("sparsity_cluster_strided_num_heads", 0)
  hparams.add_hparam("ema", True)
  hparams.add_hparam("share_qk", True)
  hparams.add_hparam("sparsity_skip_first", 0)
  hparams.add_hparam("hash_items", False)

  # Memory saving measures
  hparams.add_hparam("cache_padding_bias", True)

  # relative attention
  hparams.max_relative_position = 0
  hparams.add_hparam("local_relative", False)
  hparams.add_hparam("sparsity_cluster_relative", False)
  hparams.add_hparam("sparsity_cluster_strided_relative", False)
  hparams.add_hparam("sparsity_strided_relative", False)

  # Conditioning
  hparams.add_hparam("unconditional", True)

  return hparams


@registry.register_hparams
def sparse_imagetransformer_local2d():
  hparams = sparse_imagetransformer_base()

  # Local 2D HParams
  hparams.query_shape = (16, 16 * 3)
  hparams.memory_query_shape = (16, 16 * 3)
  hparams.memory_flange = (16, 16 * 3)

  return hparams


@registry.register_hparams
def mnist_local1d():
  """MNIST local 1D attention."""
  hparams = sparse_imagetransformer_base()

  hparams.frame_height = 28
  hparams.frame_width = 28

  hparams.local_num_heads = 4
  hparams.num_decoder_layers = 12
  hparams.query_shape = (56,)
  hparams.memory_query_shape = (56,)
  hparams.memory_flange = (56,)
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3

  return hparams


@registry.register_hparams
def mnist_full():
  """MNIST full attention."""
  hparams = mnist_local1d()

  hparams.query_shape = (784,)
  hparams.memory_query_shape = (784,)
  hparams.memory_flange(0,)

  return hparams


@registry.register_hparams
def mnist_cluster_local():
  """MNIST routing attention."""
  hparams = mnist_local1d()

  hparams.local_num_heads = 2
  hparams.sparsity_cluster_num_heads = 2
  hparams.sparsity_cluster_size = 7
  hparams.sparsity_cluster_attention_window = 112

  return hparams


@registry.register_hparams
def mnist_local_cluster_strided():
  """MNIST routing attention plus strided attention."""
  hparams = mnist_local1d()

  hparams.local_num_heads = 4
  hparams.sparsity_cluster_num_heads = 2
  hparams.sparsity_cluster_strided_num_heads = 2
  hparams.sparsity_cluster_size = 7
  hparams.sparsity_cluster_attention_window = 112

  return hparams


@registry.register_hparams
def cifar10_local1d():
  """CIFAR-10 local 1D attention."""
  hparams = sparse_imagetransformer_base()

  hparams.local_num_heads = 8
  hparams.num_decoder_layers = 12
  hparams.query_shape = (256,)
  hparams.memory_query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.hidden_size = 1024
  hparams.filter_size = 2048
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3

  return hparams


@registry.register_hparams
def cifar10_full():
  """CIFAR-10 full attention."""
  hparams = cifar10_local1d()

  hparams.query_shape = (3072,)
  hparams.memory_query_shape = (3072,)
  hparams.memory_flange = (0,)

  return hparams


@registry.register_hparams
def cifar10_local_cluster():
  """CIFAR-10 routing attention."""
  hparams = cifar10_local1d()

  hparams.local_num_heads = 4
  hparams.sparsity_cluster_num_heads = 4
  hparams.sparsity_cluster_size = 6
  hparams.sparsity_cluster_attention_window = 512

  return hparams


@registry.register_hparams
def cifar10_local_cluster_strided():
  """CIFAR-10 routing attention plus strided."""
  hparams = cifar10_local_cluster()

  hparams.local_num_heads = 4
  hparams.sparsity_cluster_num_heads = 2
  hparams.sparsity_cluster_strided_num_heads = 2

  return hparams


@registry.register_hparams
def imagenet_local1d():
  """Imagenet64 local 1D attention."""
  hparams = sparse_imagetransformer_base()

  hparams.frame_height = 64
  hparams.frame_width = 64

  hparams.hidden_size = 512
  hparams.local_num_heads = 16
  hparams.num_decoder_layers = 24
  hparams.query_shape = (512,)
  hparams.memory_query_shape = (512,)
  hparams.memory_flange = (512,)
  hparams.filter_size = 2048
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3

  return hparams


@registry.register_hparams
def imagenet_full():
  """Imagenet64 full attention."""
  hparams = imagenet_local1d()

  hparams.query_shape = (64 * 64 * 3,)
  hparams.memory_query_shape = (64 * 64 * 3,)
  hparams.memory_flange(0,)

  return hparams


@registry.register_hparams
def imagenet_local_cluster():
  """Imagenet64 routing attention."""
  hparams = imagenet_local1d()

  hparams.local_num_heads = 8
  hparams.sparsity_cluster_num_heads = 8
  hparams.sparsity_cluster_size = 12
  hparams.num_decoder_layers = 34
  hparams.sparsity_skip_first = 0
  hparams.sparsity_cluster_attention_window = 1024
  hparams.recompute_grad = True

  return hparams


@registry.register_hparams
def imagenet_local_cluster_strided():
  """Imagenet64 routing attention plus strided attention."""
  hparams = imagenet_local_cluster()

  hparams.local_num_heads = 6
  hparams.sparsity_cluster_num_heads = 5
  hparams.sparsity_cluster_strided_num_heads = 5

  return hparams
