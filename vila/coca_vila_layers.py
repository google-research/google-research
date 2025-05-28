# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""CoCa (https://arxiv.org/abs/2205.01917) Layers.

The following notations are used throughout this file:

B = batch size
N = number of tokens in encoded features
T = sequence length
D = model dims
C = decoder vocab size
"""

import math
from typing import Dict, Optional, Tuple

from flax.training import common_utils
import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes


AUX_LOSS = base_layer.AUX_LOSS
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
MetricDict = Dict[str, Tuple]
sub_config_field = base_layer.sub_config_field
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
WeightHParams = base_layer.WeightHParams
WeightInit = base_layer.WeightInit


def cross_entropy_with_logits(
    logits, targets
):
  """From t5x.losses.cross_entropy_with_logits."""
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  return loss


def compute_weighted_cross_entropy(
    logits, targets
):
  """From t5x.losses.compute_weighted_cross_entropy."""
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s targets'
        % (str(logits.shape), str(targets.shape))
    )
  vocab_size = logits.shape[-1]
  confidence = 1.0
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence)
      + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
  )
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence
  )
  total_loss = cross_entropy_with_logits(logits, soft_targets)
  total_loss = total_loss - normalizing_constant

  return jnp.sum(total_loss)


class MultimodalDecoder(base_layer.BaseLayer):
  """Transformer decoder with cross attention to encoder embeddings.

  It supports any number of unimodal transformers followed by any number of
  cross modalities transformers.

  It also supports the usage of class tokens at fprop() (not at decoding time).

  Attributes:
    num_decoder_layers: Number of decoder transformer layers.
    model_dims: Embeddding dimension.
    num_heads: Number of attention heads.
    hidden_dims: Feedforward layer hidden dim.
    decoder_vocab_size: Output target vocabulary size.
    unimodal_tr_tpl: Template. Transformer for uni modality. Only sharding
      annotations and non-essential configurations need to be provided.
    crossmodal_tr_tpl: Template. Transformer for cross modality. Only sharding
      annotations and non-essential configurations need to be provided.
  """
  model_dims: int = 0
  ff_hidden_dims: int = 0
  num_heads: int = 0
  decoder_vocab_size: int = 0
  num_class_tokens: int = 0

  num_decoder_layers: int = 0
  num_unimodal_layers: int = 0

  unimodal_tr_tpl: pax_fiddle.Config[
      layers.StackedTransformer] = template_field(layers.StackedTransformer)
  crossmodal_tr_tpl: pax_fiddle.Config[
      layers.StackedTransformer] = template_field(layers.StackedTransformer)

  def setup(self):

    ff_hidden_dims = self.ff_hidden_dims
    if ff_hidden_dims == 0:
      ff_hidden_dims = 4 * self.model_dims

    self.create_child(
        'position_emb',
        pax_fiddle.Config(
            layers.PositionalEmbedding, embedding_dims=self.model_dims))

    # Initialization of the embeddings should NOT use the default Xavier ,
    # because Xavier initialization assumes the existence of a follow-up RELU,
    # which cuts the variance by half.
    emb_init = WeightInit.Gaussian(1.0 / math.sqrt(self.model_dims))
    if self.num_class_tokens > 0:
      self.create_variable(
          'cls_emb',
          WeightHParams(
              shape=[1, self.num_class_tokens, self.model_dims],
              init=emb_init,
          ))

    self.create_child(
        'token_emb',
        pax_fiddle.Config(
            layers.Embedding,
            num_classes=self.decoder_vocab_size,
            input_dims=self.model_dims,
            params_init=emb_init,
            scale_sqrt_depth=True))

    softmax_p = pax_fiddle.Config(
        layers.FullSoftmax,
        input_dims=self.model_dims,
        num_classes=self.decoder_vocab_size)
    softmax_p.feed_forward_tpl.linear_tpl.params_init = emb_init  # pytype: disable=attribute-error
    self.create_child('softmax', softmax_p)

    # Create the unimodal transformer layers.
    if self.num_unimodal_layers > 0:
      tfm_p = self.unimodal_tr_tpl.clone().set(
          use_cross_attention=False,
          mask_self_attention=True,
          num_layers=self.num_unimodal_layers,
          model_dims=self.model_dims,
          hidden_dims=ff_hidden_dims,
          num_heads=self.num_heads,
      )

      self.create_child('unimodal_transformer', tfm_p)

      self.create_child(
          'unimodal_ln',
          pax_fiddle.Config(layers.LayerNorm, dim=self.model_dims),
      )

    # Create the crossmodal transformer layers.
    if self.num_decoder_layers > self.num_unimodal_layers:
      num_cross_modal_layers = (
          self.num_decoder_layers - self.num_unimodal_layers
      )

      tfm_p = self.crossmodal_tr_tpl.clone().set(
          use_cross_attention=True,
          mask_self_attention=True,
          num_layers=num_cross_modal_layers,
          model_dims=self.model_dims,
          hidden_dims=ff_hidden_dims,
          num_heads=self.num_heads)

      self.create_child('crossmodal_transformer', tfm_p)

      self.create_child(
          'crossmodal_ln',
          pax_fiddle.Config(layers.LayerNorm, dim=self.model_dims))

  def decoder_softmax(self, xformer_output, label_ids,
                      paddings):
    """Computes logits and softmax against the label_ids.

    Args:
      xformer_output: [B, T, D], output of fprop().
      label_ids: [B, T], labels for cross-entropy loss against logits.
      paddings: [B, T], padding of the xformer_output and label_ids.

    Returns:
      xent_output: NestedMap from softmax layer containing
        * logits: [B, T, C]. Unnormalized softmax's logits.
        * per_example_argmax: [B, T]. Argmax of an example.
        * per_example_xent: [B, T]. Cross entropy between an example's
            prediction and its label.
        * per_example_weight: [B, T]. Class_weights casted to this layer's
            dtype.
        * total_xent: A scalar. The sum of per_example_weight * per_example_xent
        * total_weight: A scalar. The sum of per_example_weight.
        * avg_xent: A scalar. total_loss / total_weight.
    """
    label_weights = (1.0 - paddings)
    xent_output = self.softmax(
        xformer_output,
        class_weights=label_weights[:, :, jnp.newaxis],
        class_ids=label_ids[:, :, jnp.newaxis])

    return xent_output

  def __call__(
      self,
      ids,
      paddings,
      encoded_features = None,
      encoded_paddings = None,
      start_time_step = 0):
    """Computes the decoding features.

    Note that the output is NOT logits. T get the actual logits, one needs to
    call self.decoder_softmax(). This is because in PAX softmax and
    cross_entropy_loss are coupled together in layers from embedding_softmax.

    Args:
      ids: [B, T]. Input ids.
      paddings: [B, T]. Paddings for both ids and labels.
      encoded_features: [B, N, D] or None. If None, fprop unimodal layers only.
      encoded_paddings: [B, N] or None. If None, all entries in encoded_features
        are considered not padded.
      start_time_step: Decode extend_step start time step. When decoding after
        prefix, start_time_step will be prefix_len - 1.

    Returns:
      cross_modalities_features: [B, T + #CLS, D],
      unimodal_features: [B, T + #CLS, D],
    """
    batch, seq_length = ids.shape

    # Setup decodings!
    if self.is_mutable_collection(base_layer.DECODE_CACHE):
      self.update_decode_state('positional_emb',
                               self.position_emb(seq_length=seq_length))
      self.update_decode_state('time_step', 0)  # pytype: disable=wrong-arg-types  # jax-ndarray

    segment_pos = jnp.tile(
        jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1]
    )

    # Create position embeddings matrix based on token positions.
    position_emb = self.position_emb(
        seq_length=seq_length, position=segment_pos
    )
    self.update_decode_state('positional_emb', position_emb)

    self.update_decode_state('time_step', start_time_step)  # pytype: disable=wrong-arg-types  # jax-ndarray

    input_emb = self.token_emb.emb_lookup(ids)
    x = input_emb + position_emb

    if self.num_class_tokens > 0:
      batch_size = ids.shape[0]
      cls_emb = jnp.tile(self.theta.cls_emb, [batch_size, 1, 1])

      cls_emb *= self.model_dims**0.5
      x = jnp.concatenate([x, cls_emb], axis=1)

      cls_paddings = jnp.zeros([batch_size, self.num_class_tokens],
                               dtype=paddings.dtype)
      paddings = jnp.concatenate([paddings, cls_paddings], axis=-1)

    # Unimodal
    if self.num_unimodal_layers > 0:
      x = self.unimodal_transformer(inputs=x, paddings=paddings,
                                    segment_pos=segment_pos)

      # Only unnormalized output is fed into crossmodal transformers.
      unimodal_output = self.unimodal_ln(x)
    else:
      unimodal_output = None

    # Cross modalities
    if (self.num_decoder_layers > self.num_unimodal_layers and
        encoded_features is not None):
      if encoded_paddings is None:
        encoded_paddings = jnp.zeros(
            shape=encoded_features.shape[:-1], dtype=encoded_features.dtype)

      x = self.crossmodal_transformer(
          inputs=x,
          paddings=paddings,
          cross_inputs=encoded_features,
          cross_paddings=encoded_paddings,
          segment_pos=segment_pos)

      x = self.crossmodal_ln(x)

    return x, unimodal_output  # pytype: disable=bad-return-type  # jax-ndarray

  def extend_step(self,
                  inputs,
                  encoded_features,
                  encoder_paddings = None,
                  segment_pos = None):
    """Compute the new output given the new token id and step number.

    This function is used by autoregressive decoding.

    Args:
      inputs: [B,]. The appended token id at index token_step.
      encoded_features: [B, N, D]. Encoder outputs.
      encoder_paddings: [B, N]. Paddings associated with encoder embeddings.
      segment_pos: Segment position of shape [B].

    Returns:
      logits: JTensor of shape [B, D] for the log-posterior.
    """
    if segment_pos is not None:
      assert segment_pos.shape == (inputs.shape[0],)

    if len(inputs.shape) == 1:
      inputs = inputs[:, jnp.newaxis]

    time_step = self.get_decode_state('time_step')
    input_emb = self.token_emb.emb_lookup(inputs[:, None])[:, 0, :]

    if segment_pos is None:
      position = jnp.zeros((inputs.shape[0], 1)) + time_step
    else:
      position = segment_pos[:, jnp.newaxis]
    position_emb = self.position_emb(seq_length=1, position=position)

    x = input_emb + position_emb

    # Unimodal
    if self.num_unimodal_layers > 0:
      x = self.unimodal_transformer.extend_step(
          inputs=x[:, 0, :], time_step=time_step, segment_pos=segment_pos)
      if self.num_decoder_layers == self.num_unimodal_layers:
        # Unimodal only
        x = self.unimodal_ln(x)

    # Crossmodal
    if self.num_decoder_layers > self.num_unimodal_layers:
      if self.num_unimodal_layers == 0:
        x = x[:, 0, :]
      if encoder_paddings is None:
        encoded_paddings = jnp.zeros(
            shape=encoded_features.shape[:-1], dtype=encoded_features.dtype)

      x = self.crossmodal_transformer.extend_step(
          inputs=x,
          time_step=time_step,
          cross_paddings=encoded_paddings,
          segment_pos=segment_pos)

      x = self.crossmodal_ln(x)

    logits = self.softmax.get_logits(x)

    self.update_decode_state('time_step', time_step + 1)

    return logits

  def transform_decode_state(
      self, transform_fn):
    """Transforms all decode state variables based on transform_fn."""

    # Unimodal
    if self.num_unimodal_layers > 0:
      self.unimodal_transformer.transform_decode_state(transform_fn)

    # Crossmodal
    if self.num_decoder_layers > self.num_unimodal_layers:
      self.crossmodal_transformer.transform_decode_state(transform_fn)


class AttenTokenPoolingLayer(base_layer.BaseLayer):
  """Attentional token pooling layer.

  Given a sequence of features, this function would extract one or more
  summary tokens via multi-head attention pooling.

  Attributes:
    input_dims: The input dimensionality.
    query_dims: Optional query/output dimensionality.
    ff_hidden_dims: The hidden dimensions of the feed-forward part of pooling
      attention.
    num_heads: The number of heads of the multi-headed attention.
    num_queries: The number of summary tokens after pooling.
    add_layer_norm: Whether to apply layer norm to the pooled tokens.
    dropout_prob: The probability of dropout on the pooled tokens.
    pool_atten_tpl: Template. Attention for pooling. Only sharding annotations
      need to be configured from here.
  """

  input_dims: int = 0
  query_dims: Optional[int] = None
  ff_hidden_dims: int = 0
  num_heads: int = 1
  num_queries: int = 1
  add_layer_norm: bool = True
  dropout_prob: float = 0.0
  pool_atten_tpl: LayerTpl = template_field(layers.DotProductAttention)

  def setup(self):
    if self.input_dims == 0:
      raise ValueError('input_dims cannot be 0.')
    query_dims = self.query_dims or self.input_dims
    ff_hidden_dims = (
        self.ff_hidden_dims if self.ff_hidden_dims > 0 else 4 * self.input_dims
    )

    input_dim = self.input_dims
    if query_dims != self.input_dims:
      input_dim = {
          'key': self.input_dims,
          'value': self.input_dims,
          'query': query_dims,
      }
    pool_atten_p = self.pool_atten_tpl.clone().set(
        name='atten',
        input_dim=input_dim,
        hidden_dim=ff_hidden_dims,
        num_heads=self.num_heads,
    )
    self.create_child('pool_attn', pool_atten_p)

    if self.add_layer_norm:
      ln_p = pax_fiddle.Config(
          layers.LayerNorm, name='pool_atten_ln', dim=query_dims
      )
      self.create_child('pool_attn_ln', ln_p)

    if self.dropout_prob > 0.0:
      dropout_p = pax_fiddle.Config(
          layers.Dropout, keep_prob=1.0 - self.dropout_prob
      )
      self.create_child('atten_dropout', dropout_p)

    query_tokens_w = WeightHParams(shape=[self.num_queries, query_dims])
    self.create_variable('pooling_attn_query', query_tokens_w)

  def __call__(
      self, tokens, paddings = None
  ):
    batch_size, seq_length = tokens.shape[:2]
    query = jnp.tile(
        self.theta.pooling_attn_query[jnp.newaxis, :, :], [batch_size, 1, 1]
    )

    if paddings is None:
      paddings = jnp.zeros([batch_size, seq_length], dtype=query.dtype)

    atten_mask = layers.convert_paddings_to_mask(paddings, dtype=paddings.dtype)

    out, _ = self.pool_attn(query, tokens, tokens, atten_mask=atten_mask)

    batch_size = out.shape[0]

    if self.add_layer_norm:
      out = self.pool_attn_ln(out)

    if self.dropout_prob > 0.0:
      out = self.atten_dropout(out)

    return out


def _pmap_concat(tensor):
  """Reduces a concatenation of the `tensor` across cores.

  Args:
    tensor: a Tensor of any shape.

  Returns:
    Reduced tensor of shape with first dimension multiplied by number of
      replicas, and other dimensions the same of rest of the input tensor.
  """
  if not base_layer.is_running_under_pmap():
    return tensor

  num_replicas = jax.device_count()

  if num_replicas <= 1:
    return tensor

  local_device_index = jax.lax.axis_index(
      axis_name=base_layer.PMAP_PARALLEL_AXIS_NAME
  )
  # Create and assign enlarged tensor based on num_replicas and replica_id.
  ext_tensor = (
      jnp.zeros([num_replicas] + list(tensor.shape), tensor.dtype)
      .at[local_device_index]
      .add(tensor)
  )

  ext_tensor = jax.lax.psum(
      ext_tensor, axis_name=base_layer.PMAP_PARALLEL_AXIS_NAME
  )

  return jnp.reshape(ext_tensor, [-1] + list(ext_tensor.shape)[2:])


class ContrastiveLossLayer(base_layer.BaseLayer):
  """Contrastive loss layer.

  Attributes:
    temperature_val: A float, temperature value of the softmax.
  """

  temperature_val: float = 1.0

  def setup(self):
    self.create_variable(
        'temperature',
        WeightHParams(shape=[], init=WeightInit.Constant(self.temperature_val)),
    )

  def __call__(self, v1, v2):
    """Computes the actual contrastive loss.

    If intraview_contrast is False, an accuracy summary will be added.

    Args:
      v1: A JTensor of shape [batch_size, dims], first embeddings
      v2: A JTensor of shape [batch_size, dims], second embeddings

    Returns:
      loss: A scalar, per batch contrastive loss.
    """
    alignment_scores = jnp.mean(jnp.sum(v1 * v2, axis=-1))
    self.add_summary('alignment_scores', alignment_scores)

    # v1: [B, D], v1_contrast: [B * N, D]
    # v2: [B, D], v2_contrast: [B * N, D]
    # labels: [B] indices in range 0 ... (B * N - 1)
    per_device_batch_size = v1.shape[0]
    if base_layer.is_running_under_pmap():
      v1_contrast = _pmap_concat(v1)
      v2_contrast = _pmap_concat(v2)

      local_device_index = jax.lax.axis_index(
          axis_name=base_layer.PMAP_PARALLEL_AXIS_NAME
      )
      labels = (
          jnp.arange(per_device_batch_size)
          + local_device_index * per_device_batch_size
      )
    else:
      # N = 1
      v1_contrast = v1
      v2_contrast = v2
      labels = jnp.arange(per_device_batch_size)

    logits_v12 = jnp.matmul(v1, jnp.transpose(v2_contrast))  # [B, B * N]
    logits_v21 = jnp.matmul(v2, jnp.transpose(v1_contrast))  # [B, B * N]

    logits_1 = logits_v12  # [B, B * N]
    logits_2 = logits_v21  # [B, B * N]

    temperature = self.theta.temperature

    # The returned losses are SUMMED across all embeddings in a batch.
    loss_1 = compute_weighted_cross_entropy(
        logits=logits_1 / temperature,
        targets=labels,
    )
    loss_2 = compute_weighted_cross_entropy(
        logits=logits_2 / temperature,
        targets=labels,
    )

    return loss_1 + loss_2
