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

"""Transformer model with sparse attention."""
import functools
import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

from sgk.sparse import connectors
from sgk.sparse import sparse_matrix
from sgk.transformer import layers


def band_and_random(length, band_size, sparsity, p=None):
  """Dense diagonal band and random off diagonal sparsity."""
  # Randomly sample from the portion of the lower triangle not
  # including the dense band. Note that zero values in the mask
  # denote the values that are kept.
  x = length - band_size
  random_triangle_size = int(x * (x + 1) / 2)

  nonzero = int(round(random_triangle_size * (1 - sparsity)))
  mask_indices = np.random.choice(
      random_triangle_size, nonzero, replace=False, p=p)
  mask = np.zeros([random_triangle_size])
  mask[mask_indices] = 1.0

  # NOTE: Numpy uses 0 in np.tri to denote the diagonal, so passing 'x'
  # will create a band of size `x + 1`.
  band = np.tri(length, length, band_size - 1).T * np.tri(length, length, 0)

  out = np.tri(length, length, -band_size)
  out[out == 1] = mask
  out += band
  out = -1e9 * (1.0 - out)
  return out


def band_and_decay(length, band_size, sparsity):
  """Dense diagonal band and random decaying off diagonal sparsity."""
  idxs = np.tril_indices(length, -band_size)

  # Weight each point by it's distance from the diagonal.
  weights = idxs[0] - idxs[1]

  # Weight decays linearly with distance from the diagonal.
  weights = np.true_divide(1.0, weights + 1e-5)

  # Normalize the weights and create the matrix.
  weights = np.true_divide(weights, np.sum(weights))
  return band_and_random(length, band_size, sparsity, p=weights)


def generate_sparse_attention_mask(sequence_length, hparams, layer):
  """Generate the sparse attention mask."""
  if hparams.sparse_attention_type == "band_and_decay":
    mask = band_and_decay(sequence_length, hparams.band_size, hparams.sparsity)
  elif hparams.sparse_attention_type == "band_and_random":
    mask = band_and_random(sequence_length, hparams.band_size, hparams.sparsity)
  else:
    raise ValueError("Unknown attention type '{}'.".format(
        hparams.sparse_attention_type))
  mask = np.reshape(
      mask.astype(np.float32), [1, 1, sequence_length, sequence_length])

  layer_name = "layer_%d" % layer
  with tf.variable_scope(layer_name):
    # NOTE: Because these patterns are random, we store them in a
    # non-trainable variable s.t. we can load the same patterns for
    # evaluation.
    out = tf.get_variable(
        "sparse_attention_mask",
        initializer=mask,
        trainable=False,
        dtype=tf.float32)
  return out


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        layer_collection=None,
                        recurrent_memory_by_layer=None,
                        chunk_number=None):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.

  Returns:
    y: a Tensors
  """
  x = decoder_input

  if hparams.sparse_attention_mode == "sparse":
    # If we want to run with our actual sparse kernels, intercept
    # the self_attention_type and replace it with our attention fn.
    seqlen = common_layers.shape_list(x)[1]
    sparse_attention_topology = sparse_matrix.SparseTopology(
        "sparse_attention", [seqlen, seqlen],
        connector=connectors.Uniform(0.955411645))  # 0.955411659
    hparams.self_attention_type = functools.partial(
        hparams.self_attention_type, topology=sparse_attention_topology)
  elif hparams.sparse_attention_mode == "masked":
    # If we're training with sparse attention, create the per-layer
    # attention bias that describes the sparsity pattern.
    #
    # NOTE: We share the same pattern across all attention heads
    # within a layer due to memory constraints (because we're not
    # actually training with sparse kernels). Per-head patterns
    # would likely perform better.
    #
    # NOTE: We also share the same pattern across all layers, as
    # protobuf can't save all of these large tensors if we create
    # more than one of them.
    decoder_self_attention_bias = generate_sparse_attention_mask(
        common_layers.shape_list(x)[1], hparams, 0)
    tf.logging.info("Generated sparse attention mask.")
  elif hparams.sparse_attention_mode == "dense":
    # Replace the dot-product attention with our memory efficient
    # version.
    hparams.self_attention_type = functools.partial(
        hparams.self_attention_type, bias=decoder_self_attention_bias)
    pass
  else:
    # For training on TPU, use T2T's standard attention.
    assert hparams.sparse_attention_mode is None

  with tf.variable_scope(name):
    for layer_idx in range(hparams.num_decoder_layers or
                           hparams.num_hidden_layers):
      x = transformer.transformer_decoder_layer(
          x,
          decoder_self_attention_bias,
          layer_idx,
          hparams,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias,
          encoder_output=encoder_output,
          cache=cache,
          decode_loop_step=decode_loop_step,
          nonpadding=nonpadding,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          losses=losses,
          layer_collection=layer_collection,
          recurrent_memory_by_layer=recurrent_memory_by_layer,
          chunk_number=chunk_number)

    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(
        x, hparams, layer_collection=layer_collection)


@registry.register_model
class SparseTransformer(transformer.Transformer):
  """A Transformer variant that supports sparse attention."""

  def __init__(self, *args, **kwargs):
    # For now, do nothing.
    super(SparseTransformer, self).__init__(*args, **kwargs)

    # Replace the decoder_function with our own.
    self._decoder_function = transformer_decoder

  @property
  def has_input(self):
    # We only use this model for decoder-only problems.
    return False


@registry.register_hparams
def sparse_transformer_imagenet64x64():
  """HParams for training image_imagenet64_gen_flat_rev."""
  hparams = transformer.transformer_big()

  hparams.num_heads = 8
  hparams.max_length = 64 * 64 * 3

  # Batch size refers to examples (not tokens).
  hparams.batch_size = 1
  hparams.shared_embedding_and_softmax_weights = False

  hparams.num_hidden_layers = 3
  hparams.attention_dropout = 0.1
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.relu_dropout = 0.1
  hparams.label_smoothing = 0.0

  ##
  ### Memory usage & TPU hparams.
  ##

  # Adafactor uses less memory than Adam. Switch to Adafactor with
  # its recommended learning rate scheme.
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000

  # Using noise broadcast in the dropout layers saves memory during training.
  hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
  hparams.relu_dropout_broadcast_dims = "1"  # length
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length

  # Avoid an expensive concat on TPU.
  hparams.symbol_modality_num_shards = 1

  hparams.add_hparam("sparse_attention_mode", "masked")
  hparams.add_hparam("sparse_attention_type", "band_and_decay")
  hparams.add_hparam("band_size", 256)
  hparams.add_hparam("sparsity", 0.95)
  return hparams


@registry.register_hparams
def fast_sparse_transformer_imagenet64x64():
  hparams = sparse_transformer_imagenet64x64()
  hparams.self_attention_type = layers.sparse_dot_product_attention
  hparams.sparse_attention_mode = "sparse"
  return hparams


@registry.register_hparams
def transformer_imagenet64x64():
  hparams = sparse_transformer_imagenet64x64()
  hparams.sparse_attention_mode = None
  return hparams


@registry.register_hparams
def fast_transformer_imagenet64x64():
  hparams = transformer_imagenet64x64()
  hparams.self_attention_type = layers.dot_product_attention
  hparams.sparse_attention_mode = "dense"
  return hparams


@registry.register_hparams
def transformer_small_imagenet64x64():
  hparams = transformer_imagenet64x64()
  hparams.num_hidden_layers = 2
  return hparams
