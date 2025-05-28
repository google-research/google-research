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

"""VILA (Vision-Lanaguage Aesthetics) model.

VILA: Learning Image Aesthetics from User Comments with Vision-Language
Pretraining (https://arxiv.org/abs/2303.14302)
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import layers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import sample_decode
from praxis.layers import activations
from praxis.layers import linears

from vila import coca_vila_layers

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
Metrics = Dict[str, tuple[JTensor, JTensor]]
WeightedScalars = pytypes.WeightedScalars
BaseInput = base_input.BaseInput
DecodeOut = base_model.DecodeOut
ProcessDecodeOut = base_model.ProcessDecodeOut
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

# Define some token ids, from the sentence piece tokenizer.
_BOS = 1
_EOS = 2
_GOOD = 391
_IMAGE = 11


def _bucketized_score_to_mos(bucketized_score):
  """Turns the bucketized score distribution to Mean Opinion Score (MOS).

  Args:
    bucketized_score: Tensor of [batch, num_buckets]. When `num_bucket>1`, each
      bucket represents the probability of score for 1 to `num_buckets`, and the
      probabilities sum up to 1. When `num_buckets=1`, this is simply the MOS
      value.

  Returns:
    Final MOS score.
  """
  num_buckets = bucketized_score.shape[-1]
  score_values = jnp.arange(1, 1 + num_buckets, dtype=jnp.float32)
  return jnp.sum(bucketized_score * score_values, axis=-1)


def _get_first_text(
    ids,
    paddings,
):
  """A util function to get the first text.

  Args:
    ids: [B, T] or [B, N, T]. Text token ids.
    paddings: [B, T] or [B, N, T]. Text token paddings.

  Returns:
    ids: [B, T]
    paddings: [B, T]
  """
  assert ids.ndim == paddings.ndim
  if ids.ndim == 3:
    ids = ids[Ellipsis, 0, :]
    paddings = paddings[Ellipsis, 0, :]
  elif ids.ndim != 2:
    raise ValueError('Only supports rank(ids) == 2 or 3')
  return ids, paddings


class CoCaVilaPretrain(base_model.BaseModel):
  """CoCa (https://arxiv.org/abs/2205.01917) for VILA pretrain.

  Attributes:
    generation_decode: Whether to enable generation decoding.
    decoding_max_len: The output sequence length to decode to.
    generative_loss_weight: The weight for generative loss.
    contrastive_loss_weight: The weight for contrastive loss.
    encoder_tpl: Template. The encoder of the images.
    decoder_tpl: Template. The decoder for the texts.
    contrastive_img_pooler_tpl: Template. Pools image features into one single
      vector.
    contrastive_loss_layer_tpl: Template. Contrastive loss between images and
      texts.
    generative_img_pooler_tpl: Template. Pools image features into features of
      same number of tokens.
    generation_decode: If True, enable generation decoding
  """

  generation_decode: bool = True
  decoding_max_len: int = -1
  decode_num_samples: int = 1
  generative_loss_weight: float = 1.0
  contrastive_loss_weight: float = 1.0
  encoder_tpl: LayerTpl = template_field(layers.VisionTransformer)
  decoder_tpl: LayerTpl = template_field(coca_vila_layers.MultimodalDecoder)
  contrastive_img_pooler_tpl: LayerTpl = template_field(
      coca_vila_layers.AttenTokenPoolingLayer
  )
  contrastive_loss_layer_tpl: LayerTpl = template_field(
      coca_vila_layers.ContrastiveLossLayer
  )
  generative_img_pooler_tpl: Optional[LayerTpl] = template_field(
      coca_vila_layers.AttenTokenPoolingLayer
  )

  def setup(self):
    """The constructor of the CoCaVilaPretrain models."""
    self.create_child('encoder', self.encoder_tpl)
    self.create_child('decoder', self.decoder_tpl)

    self.create_child('contrastive_img_pooler', self.contrastive_img_pooler_tpl)
    self.create_child('contrastive_loss', self.contrastive_loss_layer_tpl)
    self.create_child('generative_img_pooler', self.generative_img_pooler_tpl)

    next_token_sampler_p = pax_fiddle.Config(
        sample_decode.DefaultNextTokenSampler, top_k=0, top_p=1.0
    )
    self.next_token_sampler = base_layer.instantiate(next_token_sampler_p)

  def compute_image_embedding(self, input_batch):
    """Computes the image embeddings.

    It computes both contrastive and generative image embeddings.

    Args:
      input_batch: A NestedMap of the following fields:
        * image: [B, H, W, 3]. Input image.

    Returns:
      A NestedMap containing:
        * encoded_img_embed: [B, N, D], unnormalized image embedding sequence
          from vits.
        * contrastive_img_embed: [B, D], unnormalized image embedding.
        * contrastive_img_embed_norm: [B, D], normalized image embedding.
        * generative_img_embed: [B, K, D], image embedding for decoder
          cross-attention with text.
    """
    # Feeds the image input to the vision transformer.
    encoded_img_embed = self.encoder(input_batch.image)
    generative_img_embed, contrastive_img_embed = self._pool_image_embedding(
        encoded_img_embed
    )
    outputs = NestedMap()
    outputs.contrastive_img_embed = contrastive_img_embed
    outputs.contrastive_img_embed_norm = py_utils.l2_normalize(
        contrastive_img_embed
    )
    outputs.generative_img_embed = generative_img_embed

    return outputs

  def _pool_image_embedding(
      self, img_embeddings
  ):
    """Pools image level embeddings.

    Args:
      img_embeddings: [B, N, D] as image embeddings.

    Returns:
      generative_img_embed: [B, K, D]
      contrastive_img_embed: [B, D]
    """
    generative_img_embed = self.generative_img_pooler(
        img_embeddings, paddings=None
    )

    contrastive_inputs = generative_img_embed

    contrastive_img_embed = self.contrastive_img_pooler(
        contrastive_inputs, paddings=None
    )
    contrastive_img_embed = jnp.squeeze(contrastive_img_embed, axis=1)

    return generative_img_embed, contrastive_img_embed

  def compute_text_embedding(
      self, input_batch, image_encoded = None
  ):
    """Computes the text embeddings.

    It first computes self-attention only using text features, then
    cross-attention with respect to image features if image_encoded is not None.

    Args:
      input_batch: A NestedMap of the following fields:
        * ids: [B, T] or [B, 1, T]. Text token ids.
        * paddings: [B, T] or [B, 1, T]. Text token paddings.
      image_encoded: A NestedMap of the following fields or None:
        * generative_img_embed: [B, K, D]. Unnormalized image embeddings.

    Returns:
      A NestedMap contains the following fields:
        * contrastive_txt_embed: [B, D], unnormalized text embedding.
        * contrastive_txt_embed_norm: [B, D], normalized text embedding.
        * multimodal_txt_embed: [B, T, D], text embedding with cross-attention
        to image.
    """
    # Random text shuffling happens in data pipeline when there are multiple
    # texts available. Here we simply take the first text.
    ids, paddings = _get_first_text(input_batch.ids, input_batch.paddings)
    encoder_output = None
    if image_encoded:
      encoder_output = image_encoded.generative_img_embed

    multimodal_emb, unimodal_emb = self.decoder(ids, paddings, encoder_output)
    if self.decoder_tpl.num_class_tokens != 1:
      raise ValueError('Only supports number of class tokens = 1')
    # Removes the text class tokens from encoder prediction output.
    multimodal_emb = multimodal_emb[:, :-1]
    contrastive_text_embed = unimodal_emb[:, -1]
    contrastive_text_embed_norm = py_utils.l2_normalize(contrastive_text_embed)

    return NestedMap(
        multimodal_txt_embed=multimodal_emb,
        contrastive_txt_embed=contrastive_text_embed,
        contrastive_txt_embed_norm=contrastive_text_embed_norm,
    )

  def compute_predictions(self, input_batch):
    """Computes predictions from input images and texts.

    Args:
      input_batch: A NestedMap containing:
        * ids: [B, T] or [B, 1, T]. Text token ids.
        * paddings: [B, T] or [B, 1, T]. Text token paddings.
        * image: [B, H, W, 3]. Input image.

    Returns:
      A NestedMap containing
        * contrastive_img_embed: [B, D], unnormalized image embedding.
        * contrastive_img_embed_norm: [B, D], normalized image embedding.
        * generative_img_embed: [B, K, D], image embedding for decoder
        cross-attention with text.
        * contrastive_txt_embed: [B, D], unnormalized text embedding.
        * contrastive_txt_embed_norm: [B, D], normalized text embedding.
        * multimodal_txt_embed: [B, T, D], text embedding with cross-attention
        to image.
    """
    image_encoded = self.compute_image_embedding(input_batch)
    text_encoded = self.compute_text_embedding(input_batch, image_encoded)

    predictions = image_encoded
    predictions.update(text_encoded)
    return predictions

  def compute_loss(
      self,
      predictions,  # pytype: disable=signature-mismatch
      input_batch,
  ):
    """Computes loss from encoder predictions.

    Args:
      predictions: A NestedMap with the following fields
        * multimodal_txt_embed: [B, T, D]. Text features with image referred
          with cross attention.
        * contrastive_img_embed_norm: [B, D]. Normalized image embedding.
        * contrastive_txt_embed_norm: [B, D]. Normalized text embedding.
      input_batch: A NestedMap with the following fields required:
        * ids: [B, T] or [B, 1, T]. Input ids.
        * labels: [B, T] or [B, 1, T]. Target label tokens.
        * paddings: [B, T] or [B, 1, T]. Paddings for both ids and labels.

    Returns:
      A dictionary with the following items:
        'generative/loss': (per word average xent, total unpadded word counts)
        'contrastive/loss': (contrastive_loss, weight=1.0)
        'contrastive/alignment_scores': (average cosine similarity between img
          and text, weight=1.0)
        'loss': (weighted sum of generative loss and contrastive loss,
          weight=1.0)
    """
    batch_size = input_batch.ids.shape[0]

    ret_metrics = {}

    # Random text shuffling happens in data pipeline when there are multiple
    # texts available. Here we simply take the first text.
    labels, paddings = _get_first_text(input_batch.labels, input_batch.paddings)
    softmax_out = self.decoder.decoder_softmax(
        predictions.multimodal_txt_embed, labels, paddings
    )
    generative_loss = softmax_out.avg_xent
    ret_metrics['generative/loss'] = (
        generative_loss,
        softmax_out.total_weight,
    )

    # Contrastive metrics
    img_embed_norm = predictions.contrastive_img_embed_norm
    txt_embed_norm = predictions.contrastive_txt_embed_norm
    if len(txt_embed_norm.shape) == 3 and txt_embed_norm.shape[1] == 1:
      txt_embed_norm = txt_embed_norm[:, 0]

    contrastive_loss = self.contrastive_loss(img_embed_norm, txt_embed_norm) / (
        2.0 * batch_size
    )
    alignment_scores = jnp.mean(
        jnp.sum(img_embed_norm * txt_embed_norm, axis=-1)
    )
    ret_metrics['contrastive/loss'] = (contrastive_loss, batch_size)
    ret_metrics['contrastive/alignment_scores'] = (
        alignment_scores,
        batch_size,
    )

    ret_metrics['loss'] = (
        self.generative_loss_weight * generative_loss
        + self.contrastive_loss_weight * contrastive_loss,
        1.0,
    )

    return ret_metrics, {}

  def decode(
      self,
      input_batch,
      generative_img_embed = None,
  ):
    """Decodes the image from input_batch to its caption.

    Args:
      input_batch: A NestedMap containing the following fields:
        * ids: JTensor of shape [B, T] or [B, N, T], the first token of each
          batch should be 'sos', and it will be used as the first input to
          decode, N denotes number of texts.
        * paddings: JTensor of shape [B, T] or [B, N, T], padding for ids.
        * image: JTensor of shape [B, H, W, 3] where H == W.
      generative_img_embed: An optional JTensor of precomputed image embedding.
        If passed, this embedding will be reused and save computation, otherwise
        the embedding will be generated.

    Returns:
      metrics: An empty NestedMap to be consistent with base_model.BaseModel
      interface.
      results: A NestedMap containing the following fields:
        * hyp: [B, T_{decode}], decoded ids
        * hyplen: [B,], length of the valid decoded ids
        * ref: [B, T], copy from input.ids
        * reflen: [B,], length of the valid tokens in input.ids
    """
    if not self.generation_decode:
      return NestedMap(), NestedMap(), NestedMap()

    batch_size = input_batch.ids.shape[0]

    # Compute generative image embedding if not passed.
    if generative_img_embed is None:
      image_encoded = self.compute_image_embedding(input_batch)
      generative_img_embed = image_encoded.generative_img_embed

    # Use generative_img_embed after generative pooling as input to decoder.
    # Create fake input to initialize the model.
    fake_inputs = NestedMap(
        ids=jnp.zeros(
            (
                batch_size,
                self.decoding_max_len,
            ),
            dtype=input_batch.ids.dtype,
        ),
        paddings=jnp.zeros(
            (
                batch_size,
                self.decoding_max_len,
            ),
            dtype=input_batch.paddings.dtype,
        ),
        generative_img_embed=jnp.zeros_like(generative_img_embed),
    )

    self.decoder(fake_inputs.ids, fake_inputs.paddings, generative_img_embed)

    def _extend_fn(
        decoder, token, segment_pos
    ):
      with base_layer.JaxContext.new_context():
        return decoder.extend_step(
            token, generative_img_embed, segment_pos=segment_pos
        )

    def get_length_from_paddings(paddings):
      """Return the input lengths, based on the number of 0 in paddings."""
      return jnp.sum(1.0 - paddings, axis=-1).astype(jnp.int32)

    tgt_ids = jnp.array(input_batch.ids[:, 0, 0])[:, jnp.newaxis]
    tgt_paddings = jnp.array(input_batch.paddings[:, 0, 0])[:, jnp.newaxis]
    ref_ids = input_batch.ids
    ref_paddings = input_batch.paddings

    #  Greedy decoding.
    decode_out = sample_decode.greedy_decode(
        self.decoder,
        extend_step_fn=_extend_fn,
        prefix_ids=tgt_ids,
        prefix_paddings=tgt_paddings,
        seq_len=self.decoding_max_len,
        fprop_for_prefix=False,
        max_prefix_len=0,
        eos_id=_EOS,
    )  # [B, N]

    results = NestedMap(
        hyp=decode_out.output_ids,
        hyplen=decode_out.decode_lengths,
        logprobs=decode_out.logprobs,
        ref=ref_ids,
        reflen=get_length_from_paddings(ref_paddings),
        image=input_batch.image,
        prefix_ids=decode_out.prefix_ids,
        prefix_lengths=decode_out.prefix_lengths,
        generative_img_embed=generative_img_embed,
    )
    if 'image_source_id' in input_batch:
      results.image_source_id = input_batch.image_source_id
    return NestedMap(), results, NestedMap()

  def process_decode_out(
      self, input_obj, decode_out
  ):
    if not self.generation_decode:
      return NestedMap(), [], NestedMap()
    hyp = decode_out.hyp[:, 0, :]
    hyplen = decode_out.hyplen[:, 0]
    ref = decode_out.ref
    reflen = decode_out.reflen
    bs, num_texts, text_lens = decode_out.ref.shape
    ref = jnp.reshape(ref, [-1, text_lens])
    reflen = jnp.reshape(reflen, [-1])

    decoded_strs = input_obj.ids_to_strings(hyp, hyplen)
    ref_strs = input_obj.ids_to_strings(ref, reflen)  # pytype: disable=wrong-arg-types  # jnp-type
    ref_str_list = list()

    generation_outputs = []
    for batch_idx, decoded_str in enumerate(decoded_strs):
      if (
          hasattr(decode_out, 'eval_sample_weights')
          and not decode_out.eval_sample_weights[batch_idx]
      ):
        continue
      start = batch_idx * num_texts
      end = (batch_idx + 1) * num_texts
      per_example_ref_strs = ref_strs[start:end]
      ref_str_list.append(per_example_ref_strs)
      ref_str = ','.join(per_example_ref_strs)
      prefix_str = ''

      if 'image_source_id' in decode_out:
        generation_outputs.append((
            decode_out.image_source_id[batch_idx],
            {
                'decoded': decoded_str,
                'ref': ref_str,
                'prefix': prefix_str,
            },
        ))
      else:
        generation_outputs.append((
            '0',
            {
                'decoded': decoded_str,
                'ref': ref_str,
                'prefix': prefix_str,
                'image': decode_out.image,
            },
        ))
    metrics = NestedMap(
        decoded_str=[generation_outputs[i][1]['decoded'] for i in range(bs)],
        ref_str=ref_str_list,
    )

    return metrics, generation_outputs, NestedMap()


class CoCaVilaRankBasedFinetune(CoCaVilaPretrain):
  """VILA-R model (Rank-based finetuning)."""

  margin_value: float = 0.1
  bias_init: Optional[float] = 0.0
  model_dims: int = 0
  feed_forward_tpl: LayerTpl = template_field(linears.FeedForward)

  def setup(self):
    super().setup()
    residual_p = self.feed_forward_tpl.clone().set(
        input_dims=self.model_dims,
        output_dims=self.model_dims,
        activation_tpl=pax_fiddle.Config(activations.Identity),
        bias_init=self.bias_init,
    )
    self.create_child('residual_ffn', residual_p)
    # Prompt "good image".
    self._good_image_prompt_token_ids = jnp.array(
        [[_BOS, _GOOD, _IMAGE]], dtype=jnp.int32
    )
    self._good_image_prompt_token_paddings = jnp.array(
        [[0.0, 0.0, 0.0]], dtype=jnp.float32
    )

  def compute_predictions(self, input_batch):
    """Computes predictions for `input_batch`."""
    batch_size = input_batch.image.shape[0]
    predictions = super().compute_predictions(input_batch)
    image_embed = predictions.contrastive_img_embed
    image_embed = jax.lax.stop_gradient(image_embed)
    residual = self.residual_ffn(image_embed)
    image_embed += residual
    image_embed_norm = py_utils.l2_normalize(image_embed)

    prompt_text = NestedMap(
        ids=self._good_image_prompt_token_ids,
        paddings=self._good_image_prompt_token_paddings,
    )
    prompt_text_embed_norm = self.compute_text_embedding(
        prompt_text
    ).contrastive_txt_embed_norm
    prompt_text_embed_norm = jax.lax.stop_gradient(prompt_text_embed_norm)
    quality_scores = jnp.matmul(image_embed_norm, prompt_text_embed_norm.T)

    example_weights = jnp.ones([batch_size])
    if 'weight' in input_batch:
      example_weights = input_batch.weight
      if example_weights.shape != (batch_size,):
        raise ValueError(
            f'Shape of example weights should be ({batch_size},), but instead'
            f'is {example_weights.shape}'
        )
    predictions.update(
        quality_scores=quality_scores, example_weights=example_weights
    )
    return predictions

  def compute_loss(
      self,
      predictions,  # pytype: disable=signature-mismatch
      input_batch,
  ):
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (value, weight) pairs as
        values, where one of the entries is expected to correspond to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index. The base class just returns an empty dict.
    """
    # For initializing decoder parameters. Not actually using the loss here.
    super().compute_loss(predictions, input_batch)

    quality_scores = predictions.quality_scores
    example_weights = predictions.example_weights

    regression_labels = input_batch.regression_labels
    regression_labels = _bucketized_score_to_mos(regression_labels)
    regression_labels = regression_labels.reshape([-1, 1])

    quality_scores = quality_scores.reshape([-1, 1])
    pairwise_logits = quality_scores - quality_scores.T
    pairwise_labels = regression_labels - regression_labels.T
    pairwise_labels = jnp.where(pairwise_labels > 0.0, 1, 0) + jnp.where(
        pairwise_labels < 0.0, -1, 0
    )
    nonzero_indicator = jnp.where(pairwise_labels != 0, 1, 0)
    margin = nonzero_indicator * self.margin_value
    loss = jax.nn.relu(margin - pairwise_labels * pairwise_logits)
    loss = loss.sum(1).mean()

    num_valid_examples = jnp.sum(example_weights)
    metrics = NestedMap(loss=(loss, num_valid_examples))

    return metrics, {}
