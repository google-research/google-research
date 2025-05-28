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

"""Model definitions for adversarial preference learning."""

from typing import Optional
import fiddle as fdl
from flax import linen as nn
import jax
import jax.numpy as jnp
from paxml import tasks_lib
from paxml import trainer_lib
from paxml.tasks.lm.params import c4
from praxis import base_hyperparams
from praxis import base_layer
from praxis import base_model
from praxis import decoder_hparams
from praxis import decoder_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import attentions
from praxis.layers import embedding_softmax
from praxis.layers import linears
from praxis.layers import transformer_models
import utils

# Define aliases for brevity
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
sub_config_field = base_layer.sub_config_field
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
instantiate = base_hyperparams.instantiate

RANDOM = base_layer.RANDOM
PARAMS = base_layer.PARAMS
NON_TRAINABLE = base_layer.NON_TRAINABLE
DECODE_CACHE = base_layer.DECODE_CACHE
HYPER_PARAMS = base_layer.HYPER_PARAMS
PREFIX_DECODE_CACHE = base_layer.PREFIX_DECODE_CACHE

LanguageModelType = transformer_models.LanguageModelType
SampleDecoderHParams = decoder_hparams.SampleDecoderHParams


class EmbeddingOneHot(embedding_softmax.SharedEmbeddingSoftmax):
  """Computes embedding value from a one-hot encoding."""

  def emb_lookup(self, one_hot_ids):

    emb_var = jnp.transpose(self.logits_ffn.linear.theta.w)
    embs = linears.project_last_dim(one_hot_ids, emb_var)

    if self.scale_sqrt_depth:
      embs *= self.input_dims**0.5

    return embs


class EmbedTransformerLm(transformer_models.TransformerLm):
  """Transformer model which takes already-embedded inputs."""

  def __call__(self,
               inputs,
               paddings,
               labels = None,
               segment_ids = None,
               segment_pos = None,
               causal_attention_mask = None,
               start_time_step = 0):
    """Inference on a batch of inputs.

    Most of the code is equivalent to the original model. The lines which are
    different are marked with [CHANGED]
    The change is that since token weights are passed instead of just tokens,
    the shape has to change.
    It also doesn't do embedding lookup since that's assumed to be done
    previously.

    Args:
      inputs: Tensor of input embeddings for the model - shape [B, T, E]
      paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding.
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [batch, seqlen] containing weights for each target
        word. class_ids, a JTensor with shape [B, T] of int32 dtype containing
        the target class labels. class_probabilities, a JTensor with shape [B,
        T, V] of float values indicating class-membership probabilities.
      segment_ids: A JTensor of shape [B, T]. The segment that each token
        belongs to.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.
      causal_attention_mask: A JTensor of shape [B, T] where 1 indicates a token
        position with causal attention and 0 indicates bidirectional attention.
        This overrides part of the causal mask.
      start_time_step: Decode extend_step start time step. When decoding after
        prefix, start_time_step will be prefix_len.

    Returns:
      Returns xent_output, where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return. In
      addition, per_sequence_xent is added which equal to the sum of xent loss
      for tokens in a sequence.
    """

    input_emb = inputs  # [CHANGED]  Skip embedding lookup
    batch, seq_length = inputs.shape[:2]  # [CHANGED] Only use part of shape

    paddings_float32 = paddings.astype(jnp.float32)
    num_unpadded_tokens = jnp.sum(1.0 - paddings_float32)
    self.add_summary('num_unpadded_tokens', num_unpadded_tokens)
    if inputs.size != 0:
      num_tokens = jnp.array(batch * seq_length,
                             jnp.float32)  # batch * seq_length instead of size
      ratio_unpadded_tokens = num_unpadded_tokens / num_tokens
      self.add_summary('ratio_unpadded_tokens', ratio_unpadded_tokens)

    if segment_ids is None:
      assert segment_pos is None
      # Fold the paddings with the segment mask
      segment_ids = jnp.asarray(1 - paddings, jnp.int32)
      segment_pos = jnp.tile(
          jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1])

    assert self.ngrammer_tpl is None  # [CHANGED] No ngrammer

    if self.position_emb_tpl is not None:
      position_emb = self.position_emb(
          seq_length=seq_length, position=segment_pos)
      inputs = input_emb + position_emb
    else:
      inputs = input_emb

    if self.model_type == LanguageModelType.BIDIRECTIONAL:
      segment_mask = attentions.segment_mask(segment_ids, segment_ids,
                                             inputs.dtype)
    else:
      segment_mask = attentions.causal_segment_mask(segment_ids, inputs.dtype,
                                                    causal_attention_mask)

    self.update_decode_state('time_step', start_time_step)  # pytype: disable=wrong-arg-types
    output = self.transformer(
        inputs, paddings, segment_mask=segment_mask, segment_pos=segment_pos)

    # Final layer norm
    if self.final_ln_tpl is not None:
      output = self.final_ln(output)

    if self.skip_compute_loss:
      return output
    else:
      return self.compute_loss(output, labels)

  def extend_step(
      self,
      inputs,
      segment_pos = None,
      atten_mask = None,
  ):
    """Modifies the extend step function to work with a embedding input instead of tokens.
    """
    # Need to have this so our signature matches, even though we don't need it
    del atten_mask
    # Extend step should only be called with causal or prefix LM.
    assert self.model_type != LanguageModelType.BIDIRECTIONAL

    input_emb = inputs  # [CHANGED]  Skip embedding lookup
    input_emb = input_emb[:,
                          jnp.newaxis, :]  # Doesn't work right without new axis

    assert self.ngrammer_tpl is None  # [CHANGED] No ngrammer

    time_step = self.get_decode_state('time_step')
    if self.position_emb_tpl is not None:
      if segment_pos is None:
        position = jnp.zeros((inputs.shape[0], 1)) + time_step
      else:
        position = segment_pos
      position_emb = self.position_emb(seq_length=1, position=position)

      inputs = input_emb + position_emb
    else:
      inputs = input_emb

    if segment_pos is not None:
      # self.transformer expects shape [B].
      segment_pos = jnp.squeeze(segment_pos, 1)
    outputs = self.transformer.extend_step(
        inputs[:, 0, :], time_step=time_step, segment_pos=segment_pos)

    self.update_decode_state('time_step', time_step + 1)
    if self.final_ln_tpl is not None:
      outputs = self.final_ln(outputs)
    xent_output = self.compute_loss(outputs)
    return xent_output


class OneHotLM(base_model.BaseModel):
  """Wrapper class for a LM which takes one-hot embeddings."""
  model_tpl: LayerTpl = template_field(None)
  embedding_model_tpl: LayerTpl = template_field(EmbeddingOneHot)

  def setup(self):

    # Initializes the transformer
    model_p = self.model_tpl.clone()
    self.create_child('model', model_p)
    # Initializes the token embedding
    embedding_model_p = self.model.lm.softmax.hparams.clone()
    fdl.update_callable(embedding_model_p, EmbeddingOneHot)
    self.create_child('embedding_model', embedding_model_p)

  def __call__(  # pytype: disable=signature-mismatch  # jnp-array
      self, input_onehot, model_input
  ):
    predictions = self.compute_predictions(input_onehot, model_input)
    return self.model.compute_loss(predictions, model_input)

  def compute_predictions(
      self, input_onehot, model_input
  ):
    embedding = self.embedding_model.emb_lookup(input_onehot)

    model_input_embedding = model_input.copy()
    model_input_embedding.ids = embedding

    return self.model.compute_predictions(model_input_embedding)

  def simple_decode(
      self,
      input_onehot,
      use_decoded_mask,
      prefix_len,
      gs_params,
      greedy,
  ):
    """Simplified version of the decoding, which works with a one hot input.

    It's modified to be differentiable by using the Gumbel Softmax trick
    instead of sampling.

    It's simplified from praxis/sample_decode.py
    and the decode function in praxis/layers/models.py

    It's simpler because it's less configurable, and it only works when all
    prefixes are the same size.

    Args:
      input_onehot: A batch of one hot or soft onehot vectors for each sequence
        position.
      use_decoded_mask: 1 when it should use the previous output as the input. 0
        When it should use the input_onehot vector as input.
      prefix_len: The number of 0s before the first 1 in use_decoded_mask. The
        function uses forward propagation for these tokens because they all rely
        on the input_onehot.
      gs_params: Dict with 'temp' and 'hard' values.
      greedy: False to sample, and True to always use the max.

    Returns:
      A dict with:
        logits: Shape: [batch_size, input_length, vocab_size] The logits either
        predicted from the input or from the previous onehot depending on
        use_decoded_mask.
        onehot: The logits afer sampling or taking the max depending on greedy.
    """
    input_prefix_onehot = input_onehot[:, :prefix_len]
    prefix_embedding = self.embedding_model.emb_lookup(input_prefix_onehot)

    # Use forward propagation for the prefix.
    prefix_output = self.model.lm(
        prefix_embedding,
        jnp.zeros((input_onehot.shape[0], prefix_len), jnp.int32),
        start_time_step=prefix_len,
    )

    self.model.lm.transform_decode_state(
        decoder_utils.pad_state_fn(input_onehot.shape[1]))

    def extend_step_fn(model, input_onehot, segment_pos):
      embedding = model.embedding_model.emb_lookup(input_onehot)

      xent = model.model.lm.extend_step(embedding, segment_pos=segment_pos)
      return xent.logits

    # Feed the inputs which aren't part of the prefix into decoding.
    scan_xs = {
        'onehot': input_onehot[:, prefix_len:],
        'use_decoded_mask': jnp.array([use_decoded_mask[prefix_len:]])
    }

    def logits_to_onehot(logits, prng_key):
      if greedy:
        onehot = utils.calc_max_onehot(logits)
      else:
        onehot = utils.gumbel_softmax(logits, gs_params['temp'],
                                      gs_params['hard'], prng_key)
      return onehot

    def scan_body(model, carry, scan_x):
      # If use_decoded_mask, use the previous decoded value.
      # Otherwise, use the input.
      step_input_onehot = jnp.where(scan_x['use_decoded_mask'],
                                    carry['pre_out'], scan_x['onehot'])

      logits = extend_step_fn(model, step_input_onehot, None)

      onehot = logits_to_onehot(logits, model.next_prng_key())

      ys = {'onehot': onehot, 'logits': logits}
      carry = {'pre_out': onehot}

      return carry, ys

    scan_fn = nn.scan(
        scan_body,
        variable_axes={HYPER_PARAMS: 0},
        variable_broadcast=[PARAMS, NON_TRAINABLE],
        variable_carry=[DECODE_CACHE, PREFIX_DECODE_CACHE],
        split_rngs={RANDOM: True},
        in_axes=1,
        out_axes=1,
    )

    # Feed the last output from the prefix as the first decoded value.
    last_prefix_onehot = logits_to_onehot(prefix_output['logits'][:, -1],
                                          self.next_prng_key())
    _, postfix_output = scan_fn(self, {'pre_out': last_prefix_onehot}, scan_xs)

    output = {}
    output['logits'] = jnp.concatenate(
        [prefix_output['logits'], postfix_output['logits']], 1)
    output['onehot'] = jnp.concatenate(
        [last_prefix_onehot[:, None], postfix_output['onehot']], axis=1)

    return output


class C4SpmdGpt3L16AdamSmall(c4.C4SpmdGpt3L16AdamOrgHP):
  NUM_LAYERS = 24
  NUM_HEADS = 24
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = 128
  ICI_MESH_SHAPE = [1, 4, 2]


def get_regular_task_p():
  """Returns task parameters of the base experiment."""
  regular_task_p = C4SpmdGpt3L16AdamSmall().task()

  # Change some parameters to make decoding faster.
  # Pytype doesn't know about the parameters in hparams,
  # so have to disable pytype for these.
  regular_task_p.model.decoder_tpl.min_prefix_len = (
      40  # pytype: disable=attribute-error
  )
  regular_task_p.model.decoder_tpl.max_decode_steps = (
      20  # pytype: disable=attribute-error
  )
  regular_task_p.model.decoder_tpl.seqlen = (
      40  # pytype: disable=attribute-error
  )
  regular_task_p.model.decoder_tpl.fprop_for_prefix = True  # pytype: disable=attribute-error
  regular_task_p.model.lm_tpl.stacked_transformer_tpl.unroll_in_decode = False  # pytype: disable=attribute-error

  return regular_task_p


def get_onehot_task_p(dtype):
  """Returns task parameters of the onehot task."""

  class GptOneHotLM(C4SpmdGpt3L16AdamSmall):
    """GPT model with the transformer replaced to take one hot inputs."""

    def task(self):
      task_p = super().task()
      task_p.model.lm_tpl: LayerTpl

      orig_model_p = task_p.model
      orig_model_p.lm_tpl.cls = EmbedTransformerLm

      full_model_p = pax_fiddle.Config(OneHotLM, name='full_model')
      full_model_p.model_tpl = orig_model_p
      full_model_p.model_tpl.lm_tpl.stacked_transformer_tpl.unroll_in_decode = (
          False  # pytype: disable=attribute-error
      )
      full_model_p.model_tpl.fprop_dtype = (
          dtype  # pytype: disable=attribute-error
      )

      task_p.model = full_model_p

      return task_p

  return GptOneHotLM().task()


def load_model_from_checkpoint(
    checkpoint_dir,
    regular_task_p,
    dtype,
):
  """Loads model from checkpoint. Set checkpoint_dir to None to skip load."""
  regular_task = instantiate(regular_task_p)

  input_shape_dtype = jax.ShapeDtypeStruct(shape=(8, 17), dtype=jnp.int32)
  inputs_shape_dtype = NestedMap.FromNestedDict({
      'ids': input_shape_dtype,
      'labels': input_shape_dtype,
      'paddings': input_shape_dtype,
      'segment_ids': input_shape_dtype,
      'segment_pos': input_shape_dtype,
      'weights': input_shape_dtype
  })
  if checkpoint_dir is not None:
    vars_weight_params = regular_task.model.abstract_init_with_metadata(
        inputs_shape_dtype)
    train_state_global_shapes = regular_task.create_train_state_padded_shapes(
        vars_weight_params)
    train_state_global_shapes = train_state_global_shapes.replace(opt_states=[])
    model_states = tasks_lib.restore_pmap_from_tensorstore(
        train_state_global_shapes, checkpoint_dir
    )
  else:
    model_states = trainer_lib.initialize_model_state(
        regular_task,
        jax.random.PRNGKey(6849),
        inputs_shape_dtype,
        do_init_checkpoint_rules=False,
        discard_opt_states=True,
    )[0]

  model_states = jax.tree_util.tree_map(
      lambda x: jnp.array(x, dtype=dtype), model_states
  )
  return model_states
