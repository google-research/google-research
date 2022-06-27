# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Routing Transformer for text problems."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from routing_transformer import utils


@registry.register_model
class SparseTransformer(t2t_model.T2TModel):
  """Sparse Transformer model."""

  def vocab_size(self):
    return self.problem_hparams.vocab_size["targets"]

  @property
  def batch_size(self):
    if self.is_decode:
      # Divide by num_cores for TPU decoding, e.g. 2x2 DF = 8 cores
      num_cores = self.hparams.num_decode_cores
      return self._decode_hparams.batch_size // num_cores
    else:
      return self.hparams.batch_size

  @property
  def hidden_size(self):
    return self.hparams.hidden_size

  @property
  def num_decoder_layers(self):
    return self.hparams.num_decoder_layers

  @property
  def num_encoder_layers(self):
    return self.hparams.num_encoder_layers

  @property
  def is_decode(self):
    return self.hparams.mode == tf_estimator.ModeKeys.PREDICT

  def process_partial_targets_decoding(self, targets):
    """Processes partially generated targets in decoding mode."""
    targets_shape = utils.shape_list(targets)
    seq_length = targets_shape[1]
    blocks_per_dim = [
        s // q for s, q in zip([seq_length], self.hparams.query_shape)
    ]
    targets = tf.reshape(
        targets, [targets_shape[0], -1,
                  np.prod(self.hparams.query_shape), 1])

    targets = utils.unflatten_blocks_nd(targets, blocks_per_dim)
    targets = utils.put_back_blocks_nd(targets, self.hparams.query_shape)
    targets = tf.reshape(targets, [-1, seq_length])
    return targets

  def prepare_decoder(self, targets):
    """Prepares targets for transformer decoder."""
    shape = utils.shape_list(targets)
    # sequence should be [batch, seq_length]
    assert len(shape) == 2, "Sequence tensors should be 2-dimensional"
    assert len(
        self.hparams.query_shape) == 1, "query shape should be 1-dimensional"

    # Mask random positions
    if self.hparams.target_dropout:
      targets = tf.where(
          tf.random.uniform(shape) < self.hparams.target_dropout,
          tf.zeros_like(targets), targets)
    # Shift positions
    targets = tf.expand_dims(targets, axis=-1)
    targets = utils.right_shift_blockwise_nd(targets, self.hparams.query_shape)
    targets = tf.squeeze(targets, axis=-1)
    # Add token embeddings
    targets = utils.get_embeddings(
        targets=targets,
        hidden_size=self.hparams.embedding_dims,
        vocab_size=self.vocab_size)
    if self.hparams.dropout:
      targets = tf.nn.dropout(targets, 1 - self.hparams.dropout)
    targets = tf.layers.dense(
        targets, self.hidden_size, activation=None, name="emb_dense")
    if self.hparams.add_timing_signal:
      targets += utils.get_timing_signal_1d(self.hparams.max_target_length,
                                            self.hidden_size)
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
    if self.hparams.embedding_dims != self.hparams.hidden_size:
      decoder_output = tf.layers.dense(
          decoder_output,
          self.hparams.embedding_dims,
          activation=None,
          name="final_dense_1")
    output = tf.layers.dense(
        decoder_output, self.vocab_size, activation=None, name="final_dense_2")
    return output

  def body(self,
           features,
           decode_step=None,
           cache=None,
           decoding_stats=None,
           add_summary=True):
    encoder_output = None
    extra_losses = []
    padding_bias = None
    if not self.hparams.fast_decode:
      decode_step = None
    if "inputs" in features:
      inputs = features["inputs"]
      # remove the last two dimensions that are always 1.
      inputs = tf.reshape(inputs,
                          utils.shape_list(inputs)[:2] + [self.hidden_size])
      # Padding bias only used for seq2seq models.
      padding_bias = utils.embedding_to_padding(inputs)
      # Mask random positions
      shape = utils.shape_list(inputs)
      if self.hparams.input_dropout:
        inputs = tf.where(
            tf.random.uniform(shape) < self.hparams.input_dropout,
            tf.zeros_like(inputs), inputs)
      if self.hparams.add_timing_signal:
        inputs += utils.get_timing_signal_1d(self.hparams.max_length,
                                             self.hidden_size)
      if cache is not None and -1 in cache:
        encoder_output = cache[-1]
      else:
        encoder_output = utils.transformer_encoder_layers(
            inputs=inputs,
            num_layers=self.num_encoder_layers,
            hparams=self.hparams,
            losses=extra_losses,
            name="encoder",
            token_bias=features.get("token_bias_inputs"),
            padding_bias=padding_bias)
      if cache is not None and -1 not in cache:
        cache[-1] = encoder_output
    targets = tf.to_int32(features["targets"])
    # remove the last two dimensions that are always 1.
    targets = tf.reshape(targets, utils.shape_list(targets)[:2])
    # Clamp targets to max_target_length
    targets = targets[:, :self.hparams.max_target_length]
    if self.is_decode:
      targets = self.process_partial_targets_decoding(targets)
    decoder_input = self.prepare_decoder(targets)

    decoder_output = utils.transformer_decoder_layers(
        inputs=decoder_input,
        num_layers=self.num_decoder_layers,
        hparams=self.hparams,
        encoder_output=encoder_output,
        decode_step=decode_step,
        losses=extra_losses,
        cache=cache,
        name="decoder",
        decoding_stats=decoding_stats,
        token_bias_inputs=features.get("token_bias_inputs"),
        token_bias_targets=features.get("token_bias_targets"),
        padding_bias=padding_bias)
    logits = self.produce_output(decoder_output)

    # Return logits as-is in decoding mode
    if self.is_decode:
      return logits

    # Add cross entropy loss
    one_hot_targets = tf.one_hot(
        tf.cast(targets, dtype=tf.int32), self.vocab_size)
    x_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=one_hot_targets, logits=logits)
    weights = tf.to_float(tf.not_equal(targets, 0))
    loss = tf.reduce_sum(x_entropy * weights) / tf.reduce_sum(weights)
    if add_summary:
      tf.summary.scalar("losses/weight", tf.reduce_sum(weights))
      tf.summary.scalar("losses/x_entropy", tf.reduce_sum(x_entropy * weights))

    loss_dict = {"training": loss}
    if extra_losses:
      loss_dict["extra_loss"] = tf.add_n(extra_losses)
    # hack for T2T metrics
    logits = tf.reshape(
        logits,
        utils.shape_list(logits)[:2] + [1, 1] + utils.shape_list(logits)[-1:])
    return logits, loss_dict

  def sample(self, features, decode_step, cache, decoding_stats):
    """Sample step for infer."""
    with tf.variable_scope("sparse_transformer/body", reuse=tf.AUTO_REUSE):
      logits = self.body(features, decode_step, cache, decoding_stats)
      if not self.hparams.fast_decode:
        logits = tf.gather(logits, decode_step, axis=1)
      logits = tf.reshape(logits, [self.batch_size, self.vocab_size])
      # Should not use top_k and top_p together
      assert (self.hparams.sampling_keep_top_k *
              (1 - self.hparams.nucleus_sampling) == 0)
      if self.hparams.sampling_keep_top_k:
        tf.logging.info("Top-k sampling top_k = {}".format(
            self.hparams.sampling_keep_top_k))
        values, _ = tf.math.top_k(logits, k=self.hparams.sampling_keep_top_k)
        k_largest = tf.reduce_min(values)
        logits = tf.where(tf.less_equal(logits, k_largest),
                          tf.ones_like(logits)*-1e9, logits)
      if self.hparams.nucleus_sampling < 1:
        logits = self.nucleus_sampling(logits)
      sample = self.multinomial_squeeze(logits, self.hparams.sampling_temp)
      sample = tf.reshape(sample, [self.batch_size])
      return sample, logits

  def nucleus_sampling(self, logits):
    """Nucleus sampling."""
    p = self.hparams.nucleus_sampling
    tf.logging.info("Nucleus sampling top_p = {}".format(p))
    sort_indices = tf.argsort(logits, axis=-1, direction="DESCENDING")
    probs = tf.gather(tf.nn.softmax(logits), sort_indices, batch_dims=1)
    cumprobs = tf.cumsum(probs, axis=-1, exclusive=True)
    # The top 1 candidate always will not be masked.
    # This way ensures at least 1 indices will be selected.
    sort_mask = tf.cast(tf.greater(cumprobs, p), logits.dtype)
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(logits.shape[0]), axis=-1),
        [1, logits.shape[1]])
    top_p_mask = tf.scatter_nd(
        tf.stack([batch_indices, sort_indices], axis=-1), sort_mask,
        logits.shape)
    logits -= top_p_mask * logits.dtype.max
    return logits

  def infer(self, features, **kwargs):
    with tf.variable_scope("sparse_transformer", reuse=tf.AUTO_REUSE):
      features = self.bottom(features)
    decode_length = self.hparams.max_target_length
    cache = {}
    decoding_stats = {}
    targets_old = features.get("targets")
    start_step = 0
    initial_output = tf.zeros((self.batch_size, decode_length, 1, 1),
                              dtype=tf.int32)
    initial_logits = tf.zeros((self.batch_size, decode_length, self.vocab_size))

    # call body once to initialize cache with representations of input frames.
    features["targets"] = initial_output
    # Set shape of inputs
    if "inputs" in features:
      features["inputs"].set_shape([self.batch_size,
                                    self.hparams.max_length,
                                    1,
                                    self.hparams.hidden_size])
    with tf.variable_scope("sparse_transformer/body", reuse=tf.AUTO_REUSE):
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
          indices=[[b, pos, 0, 0] for b in range(self.batch_size)],
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
        [start_step, initial_output, initial_logits, cache, decoding_stats],
        back_prop=False,
        parallel_iterations=1)

    original_shape = [decode_length]

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


def target_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # Unused
  return x


@registry.register_hparams
def sparse_transformer_local():
  """Set of hyperparameters for a sparse model using only local."""
  hparams = common_hparams.basic_params1()
  hparams.max_length = 4096
  hparams.batch_size = 4096
  hparams.add_hparam("max_target_length", 4096)
  hparams.add_hparam("add_timing_signal", False)
  hparams.add_hparam("local_num_heads", 8)
  hparams.add_hparam("sparsity_cluster_num_heads", 0)
  hparams.add_hparam("sparsity_strided_num_heads", 0)
  hparams.add_hparam("sparsity_cluster_strided_num_heads", 0)
  hparams.add_hparam("sparsity_skip_first", 0)
  hparams.add_hparam("ema", True)
  hparams.add_hparam("query_shape", (512,))
  hparams.add_hparam("memory_query_shape", (512,))
  hparams.add_hparam("memory_flange", (512,))
  hparams.add_hparam("sparsity_cluster_size", 0)
  hparams.add_hparam("sparsity_cluster_attention_window", 0)
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 24)
  hparams.add_hparam("attention_key_channels", 0)  # Uses hidden_size
  hparams.add_hparam("attention_value_channels", 0)  # Uses hidden_size
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("ffn_layer", "geglu")
  hparams.add_hparam("filter_size", 2048)  # Used in ffn_layer
  hparams.add_hparam("relu_dropout", 0.0)  # Used in ffn_layer
  hparams.add_hparam("input_dropout", 0.0)  # dropout on input sequences
  hparams.add_hparam("target_dropout", 0.0)  # dropout on target sequences
  hparams.add_hparam("use_tpu", True)
  hparams.tpu_enable_host_call = True  # Enable summaries on TPU
  hparams.pad_batch = True
  hparams.bottom = {
      "targets": target_bottom,
  }

  # Optimizer
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
  hparams.summarize_vars = True
  hparams.hidden_size = 512

  # Memory saving measures
  hparams.add_hparam("cache_padding_bias", False)
  hparams.add_hparam("embedding_dims", 512)
  hparams.add_hparam("share_qk", True)
  hparams.shared_embedding = True
  hparams.shared_embedding_and_softmax_weights = True

  # relative attention
  hparams.max_relative_position = 1024
  hparams.add_hparam("local_relative", True)
  hparams.add_hparam("sparsity_cluster_relative", True)
  hparams.add_hparam("sparsity_cluster_strided_relative", True)
  hparams.add_hparam("sparsity_strided_relative", False)

  # Decoding
  hparams.add_hparam("nucleus_sampling", 0.9)
  hparams.add_hparam("num_decode_cores", 8)
  hparams.add_hparam("fast_decode", False)

  # Clustering hparams
  hparams.add_hparam("beta", 1e-4)
  hparams.add_hparam("decay", 0.999)

  # LSH attention as in Reformer
  hparams.add_hparam("hash_items", False)

  hparams.add_hparam("token_bias_wt_trainable", False)
  return hparams


@registry.register_hparams
def enwik8_local8k():
  """Local attention on sequence length 8k."""
  hparams = sparse_transformer_local()

  hparams.max_length = 8192
  hparams.batch_size = 8192
  hparams.hidden_size = 1024
  hparams.embedding_dims = 1024
  hparams.num_decoder_layers = 22
  hparams.local_num_heads = 8
  hparams.filter_size = 3072
  hparams.query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.max_relative_position = 513
  hparams.attention_dropout = 0.4
  hparams.relu_dropout = 0.4

  return hparams


@registry.register_hparams
def enwik8_local_cluster8k():
  """Routing attention on sequence length 8k."""
  hparams = enwik8_local8k()

  hparams.sparsity_cluster_size = 16
  hparams.local_num_heads = 4
  hparams.sparsity_cluster_num_heads = 4
  hparams.sparsity_cluster_attention_window = 512
  hparams.num_decoder_layers = 12
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0

  return hparams


@registry.register_hparams
def enwik8_local12k():
  """Local attention on sequence length 12k."""
  hparams = enwik8_local8k()

  hparams.max_length = 12288
  hparams.batch_size = 12288
  hparams.query_shape = (128,)
  hparams.memory_flange = (128,)
  hparams.attention_dropout = 0.4
  hparams.relu_dropout = 0.4
  hparams.max_relative_position = 257
  hparams.hidden_size = 1024
  hparams.embedding_dims = 1024

  return hparams


@registry.register_hparams
def enwik8_local_cluster12k():
  """Routing attention on sequence length 12k."""
  hparams = enwik8_local12k()

  hparams.sparsity_cluster_size = 48
  hparams.local_num_heads = 4
  hparams.sparsity_cluster_num_heads = 4
  hparams.sparsity_cluster_attention_window = 256
  hparams.num_decoder_layers = 12
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0

  return hparams


@registry.register_hparams
def wikitext103_local4k():
  """Local attention on sequence length 4k."""
  hparams = sparse_transformer_local()
  hparams.max_length = 4096
  hparams.batch_size = 4096
  hparams.hidden_size = 1024
  hparams.embedding_dims = 256
  hparams.filter_size = 3072
  hparams.local_num_heads = 16
  hparams.num_decoder_layers = 14
  hparams.query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.attention_dropout = 0.4
  hparams.relu_dropout = 0.4
  hparams.dropout = 0.4
  hparams.target_dropout = 0.25
  hparams.max_relative_position = 513
  hparams.weight_decay = 2e-5
  return hparams


@registry.register_hparams
def wikitext103_local_cluster4k():
  """Routing attention on sequence length 4k."""
  hparams = wikitext103_local4k()
  hparams.local_num_heads = 8
  hparams.sparsity_cluster_num_heads = 8
  hparams.sparsity_cluster_attention_window = 512
  hparams.sparsity_cluster_size = 8
  hparams.share_qk = True
  return hparams


@registry.register_hparams
def wikitext103_local_hash4k():
  """LSH plus local attention on sequence length 4k."""
  hparams = wikitext103_local_cluster4k()
  hparams.hash_items = True
  return hparams


@registry.register_hparams
def wikitext103_local_cluster1k():
  """Routing attention on sequence length 1k."""
  hparams = wikitext103_local_cluster4k()
  hparams.max_length = 1024
  hparams.batch_size = 1024
  hparams.sparsity_cluster_size = 2
  hparams.num_decoder_layers = 20
  hparams.target_dropout = 0.2
  hparams.weight_decay = 2e-5
  hparams.share_qk = True
  return hparams


@registry.register_hparams
def wikitext103_local_hash1k():
  """LSH plus local attention on sequence length 1k."""
  hparams = wikitext103_local_cluster1k()
  hparams.hash_items = True
  return hparams


@registry.register_hparams
def pg19_local8k():
  """Local attention on sequence length 8k."""
  hparams = wikitext103_local4k()
  hparams.max_length = 8192
  hparams.batch_size = 8192
  hparams.max_target_length = 8192
  hparams.hidden_size = 1032
  hparams.embedding_dims = 1032
  hparams.filter_size = 4096
  hparams.local_num_heads = 8
  hparams.sparsity_cluster_num_heads = 0
  hparams.num_decoder_layers = 24
  hparams.query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.dropout = 0.0
  hparams.target_dropout = 0.0
  hparams.max_relative_position = 513
  hparams.weight_decay = 0
  return hparams


@registry.register_hparams
def pg19_local_cluster8k():
  """Routing attention on sequence length 8k."""
  hparams = pg19_local8k()
  hparams.local_num_heads = 6
  hparams.sparsity_cluster_num_heads = 2
  hparams.num_decoder_layers = 22
  hparams.sparsity_skip_first = 21
  hparams.sparsity_cluster_size = 16
  hparams.sparsity_cluster_attention_window = 512
  hparams.max_relative_position = 513
  return hparams


@registry.register_hparams
def meena_local2k():
  """Hparams for Meena local attention model."""
  hparams = sparse_transformer_local()
  hparams.max_length = 2048
  hparams.batch_size = 2048
  hparams.max_target_length = 512
  hparams.hidden_size = 1024
  hparams.embedding_dims = 256
  hparams.num_encoder_layers = 6
  hparams.num_decoder_layers = 15
  hparams.local_num_heads = 8
  hparams.filter_size = 3072
  hparams.query_shape = (256,)
  hparams.memory_query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.max_relative_position = 513
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  return hparams


@registry.register_hparams
def meena_local_cluster2k():
  """Hparams for Meena local attention model."""
  hparams = sparse_transformer_local()
  hparams.max_length = 2048
  hparams.batch_size = 2048
  hparams.max_target_length = 512
  hparams.hidden_size = 1024
  hparams.embedding_dims = 512
  hparams.num_encoder_layers = 6
  hparams.num_decoder_layers = 15
  hparams.local_num_heads = 6
  hparams.sparsity_skip_first = 5
  hparams.sparsity_cluster_num_heads = 2
  hparams.sparsity_cluster_size = 8
  hparams.sparsity_cluster_attention_window = 512
  hparams.filter_size = 3072
  hparams.query_shape = (256,)
  hparams.memory_query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.max_relative_position = 513
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.ema = True
  return hparams


@registry.register_hparams
def arxiv_local24k():
  """Hparams for Meena local attention model."""
  hparams = sparse_transformer_local()
  hparams.max_length = 24000
  hparams.batch_size = 24000
  hparams.max_target_length = 512
  hparams.hidden_size = 256
  hparams.embedding_dims = 256
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 8
  hparams.local_num_heads = 8
  hparams.filter_size = 3072
  hparams.query_shape = (256,)
  hparams.memory_query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.max_relative_position = 513
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  return hparams


@registry.register_hparams
def arxiv_local_cluster24k():
  """Hparams for Meena local attention model."""
  hparams = sparse_transformer_local()
  hparams.max_length = 24064
  hparams.batch_size = 24064
  hparams.max_target_length = 512
  hparams.hidden_size = 1024
  hparams.embedding_dims = 1024
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 8
  hparams.local_num_heads = 6
  hparams.sparsity_cluster_num_heads = 2
  hparams.sparsity_cluster_size = 47
  hparams.sparsity_cluster_attention_window = 512
  hparams.filter_size = 4096
  hparams.query_shape = (256,)
  hparams.memory_query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.max_relative_position = 257
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.ema = True
  return hparams


@registry.register_hparams
def pubmed_local10k():
  """Hparams for Meena local attention model."""
  hparams = sparse_transformer_local()
  hparams.max_length = 10240
  hparams.batch_size = 10240
  hparams.max_target_length = 512
  hparams.hidden_size = 256
  hparams.embedding_dims = 256
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 10
  hparams.local_num_heads = 8
  hparams.filter_size = 3072
  hparams.query_shape = (128,)
  hparams.memory_query_shape = (128,)
  hparams.memory_flange = (128,)
  hparams.max_relative_position = 257
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.input_dropout = 0.1
  return hparams


@registry.register_hparams
def pubmed_local_cluster10k():
  """Hparams for Meena local attention model."""
  hparams = sparse_transformer_local()
  hparams.max_length = 10240
  hparams.batch_size = 10240
  hparams.max_target_length = 512
  hparams.hidden_size = 256
  hparams.embedding_dims = 256
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 10
  hparams.local_num_heads = 4
  hparams.sparsity_cluster_num_heads = 4
  hparams.sparsity_cluster_size = 40
  hparams.sparsity_skip_first = 2
  hparams.sparsity_cluster_attention_window = 256
  hparams.filter_size = 3072
  hparams.query_shape = (128,)
  hparams.memory_query_shape = (128,)
  hparams.memory_flange = (128,)
  hparams.max_relative_position = 257
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.input_dropout = 0.1
  return hparams


@registry.register_hparams
def translate_local_cluster512():
  """Routing attention on sequence length 512 with transformer_big hparams."""
  hparams = pg19_local_cluster8k()
  hparams.max_length = 512
  hparams.max_target_length = 512
  hparams.batch_size = 8192
  hparams.hidden_size = 1024
  hparams.embedding_dims = 1024
  hparams.share_qk = True
  hparams.local_num_heads = 14
  hparams.sparsity_cluster_num_heads = 2
  hparams.num_encoder_layers = 6
  hparams.num_decoder_layers = 6
  hparams.sparsity_skip_first = 4
  hparams.sparsity_cluster_size = 1
  hparams.query_shape = (256,)
  hparams.memory_query_shape = (256,)
  hparams.memory_flange = (256,)
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.dropout = 0.0
  hparams.input_dropout = 0.0
  hparams.filter_size = 4096
  hparams.sparsity_cluster_attention_window = 512
  hparams.max_relative_position = 513
  hparams.weight_decay = 0
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.optimizer = "adam"
  hparams.learning_rate_schedule = "legacy"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.ema = True
  return hparams


@registry.register_hparams
def translate_local_cluster1k():
  """Routing attention on sequence length 1k with transformer_big hparams."""
  hparams = translate_local_cluster512()
  hparams.max_length = 1024
  hparams.max_target_length = 1024
  hparams.sparsity_cluster_size = 2
  return hparams


@registry.register_hparams
def translate_local_cluster2k():
  """Routing attention on sequence length 2k with transformer_big hparams."""
  hparams = translate_local_cluster512()
  hparams.max_length = 2048
  hparams.max_target_length = 2048
  hparams.sparsity_cluster_size = 4
  return hparams


@registry.register_hparams
def translate_local_cluster4k():
  """Routing attention on sequence length 4k with transformer_big hparams."""
  hparams = translate_local_cluster512()
  hparams.max_length = 4096
  hparams.max_target_length = 4096
  hparams.sparsity_cluster_size = 8
  return hparams


@registry.register_hparams
def translate_local_cluster8k():
  """Routing attention on sequence length 8k with transformer_big hparams."""
  hparams = translate_local_cluster512()
  hparams.max_length = 8192
  hparams.max_target_length = 8192
  hparams.sparsity_cluster_size = 16
  return hparams


@registry.register_hparams
def translate_local8k():
  """Local attention on sequence length 8k with transformer_big hparams."""
  hparams = translate_local_cluster8k()
  hparams.share_qk = False
  hparams.local_num_heads = 16
  hparams.sparsity_cluster_num_heads = 0
  return hparams
