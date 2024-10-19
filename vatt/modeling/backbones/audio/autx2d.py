# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Model defination for the AuTx2D Audio Model."""

import tensorflow as tf

from vatt.modeling.common import transformers


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class AuTx2D(tf.keras.layers.Layer):
  """The general Transformer head for pre-training different modalities."""

  def __init__(self,
               # patching parameters
               temporal_patch_size=4,
               spectoral_patch_size=16,
               # pre-projection parameters
               pre_projection=False,
               # transformer parameters
               d_model=768,
               d_kv=64,
               d_ff=3072,
               num_layers=2,
               num_heads=12,
               pre_norm=False,
               use_bias=True,
               activation="gelu",
               dropout_rate=0.1,
               layer_norm_epsilon=1e-6,
               # masking parameters
               use_masking=True,
               mask_rate=0.2,
               # positional embedding parameters
               max_temporal_buckets=16,
               max_spectoral_buckets=4,
               # post-projection parameters
               post_projection=False,
               d_post_proj=1024,
               post_proj_activation="gelu",
               name="au_tx_2d",
               **kwargs):
    super(AuTx2D, self).__init__(name=name)
    self.pre_projection = pre_projection
    self.post_projection = post_projection
    self.d_model = d_model
    # masking parameters
    self.use_masking = use_masking
    self.mask_rate = mask_rate

    # define voxel to patch module
    patch_stack = (temporal_patch_size,
                   spectoral_patch_size)
    self.spectrum_to_patch = tf.keras.layers.Conv2D(
        filters=d_model,
        kernel_size=patch_stack,
        strides=patch_stack,
        padding="valid",
        name="spectrum_to_patch",
        )

    if self.pre_projection:
      self.pre_proj = tf.keras.layers.Dense(
          d_model,
          activation=activation,
          name="pre_tx_projection",
          )
    else:
      self.pre_proj = tf.identity

    # define positional embedding module
    self.pos_embedding_lookup = transformers.SpectroTemporalEmbeddings(
        hidden_size=self.d_model,
        max_temporal_buckets=max_temporal_buckets,
        max_spectoral_buckets=max_spectoral_buckets,
        )

    # define transformer head
    self.tx = transformers.TransformerEncoder(
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        pre_norm=pre_norm,
        use_bias=use_bias,
        activation=activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        name="transformer",
        )

    if self.post_projection:
      self.post_proj = tf.keras.layers.Dense(
          d_post_proj,
          activation=post_proj_activation,
          name="post_tx_projection",
          )
    else:
      self.post_proj = tf.identity

  def build(self, input_shapes):
    token_embds_kwargs = {
        "shape": (self.d_model,),
        "initializer": tf.keras.initializers.get("glorot_normal"),
        "trainable": True,
        "dtype": tf.float32,
    }
    if self.use_masking:
      # define mask_token_embd as a learnable vector
      self.mask_token_embd = self.add_weight(
          name="mask_embedding",
          **token_embds_kwargs,
          )

    # add special token Aggregator
    self.agg_token_embd = self.add_weight(
        name="agg_embedding",
        **token_embds_kwargs,
        )

  def random_embd_mask(self, input_embds, input_attn_mask=None):
    """Replacing input tokens with mask_embds, random_embds or nothing.

    Args:
      input_embds: input sequence of token embeddings
      input_attn_mask: padding/attention mask for input sequence

    Returns:
      input_embds: given input (unchanged - for loss purposes)
      input_attn_mask: given padding/attention mask (unchanged)
      masked_input_embds: masked inputs according to both padding/attention mask
        and randomly generated token masks
      mask_pos: a sequence with same shape as input, containing 0/1 in
        locations where input tokens have been manipulated (1) or unchanged (0)
    """

    batch_size, seq_len, embd_dim = get_shape(input_embds)
    if input_attn_mask is None:
      input_attn_mask = tf.ones((batch_size, 1), dtype=tf.int32)
    # initialize placers for random ids
    mask_ids = tf.zeros((batch_size * seq_len,), dtype=tf.int32)
    random_ids = tf.zeros((batch_size * seq_len,), dtype=tf.int32)
    no_touch_ids = tf.zeros((batch_size * seq_len,), dtype=tf.int32)
    # control where to mask
    randomness = tf.random.uniform((batch_size * seq_len, 3))
    # a random set of token embeddings to be used as 10% of masked token embds
    embds_flattened = tf.stop_gradient(
        tf.reshape(input_embds, [-1, embd_dim])
        )
    shuffled_token_embds = tf.gather(
        embds_flattened,
        tf.random.shuffle(tf.range(tf.shape(embds_flattened)[0]))
        )
    shuffled_token_embds = tf.reshape(
        shuffled_token_embds,
        [batch_size, seq_len, embd_dim]
        )
    # fill in the placers where to mask
    for n in range(batch_size*seq_len):
      if randomness[n, 0] <= self.mask_rate:
        # do masking
        where_to_mask = tf.sparse.SparseTensor(
            indices=[[n]], values=[1], dense_shape=(batch_size * seq_len,))

        if randomness[n, 1] <= 0.8:
          # 80% mask
          mask_ids += tf.sparse.to_dense(where_to_mask)

        elif randomness[n, 2] <= 0.5:
          # 10% replace with random token from random set of tokens
          random_ids += tf.sparse.to_dense(where_to_mask)

        else:
          # 10% do nothing, but keep track of it
          no_touch_ids += tf.sparse.to_dense(where_to_mask)

    # get the masks tensor containing 0/1s indicating where to replace with
    # self.mask_token_embd, a learnable vector
    masks = tf.reshape(tf.stack(mask_ids), [batch_size, seq_len])
    # get the masks tensor containing 0/1s indicating where to replace with
    # a randomly chosen token from the current sequence (across all batches)
    randoms = tf.reshape(tf.stack(random_ids), [batch_size, seq_len])
    # find where the token was unchanged but it was flagged as mask
    no_touches = tf.reshape(tf.stack(no_touch_ids), [batch_size, seq_len])
    # apply the attention/padding mask to all the masks
    masks = tf.cast(masks * input_attn_mask, tf.float32)
    randoms = tf.cast(randoms * input_attn_mask, tf.float32)
    no_touches = tf.cast(no_touches * input_attn_mask, tf.float32)
    # replace the location of resulting masks with the mask values
    # (mask_token_embd / shuffled_token_embds)
    masked_input_embds = (
        input_embds * (1-masks-randoms)[:, :, None] +
        self.mask_token_embd[None, None, :] * masks[:, :, None] +
        shuffled_token_embds * randoms[:, :, None]
        )
    # add random shuffle and untouched locations to the mask locations
    mask_pos = masks + randoms + no_touches

    return masked_input_embds, mask_pos

  def _flatten_inputs(self,
                      inputs):

    input_shape = get_shape(inputs)
    bs = input_shape[0]
    d_embd = input_shape[-1]

    inputs = tf.reshape(inputs, [bs, -1, d_embd])
    return inputs, input_shape

  def _append_special_tokens(self,
                             inputs):

    batch_size = get_shape(inputs)[0]

    # (1, 1, d_model)
    special_embds = self.agg_token_embd[None, None, :]

    # (batch_size, 1, d_model)
    special_embds = tf.tile(special_embds, [batch_size, 1, 1])

    return tf.concat([special_embds, inputs], axis=1)

  def call(self,
           inputs,
           training=False):

    # spectrum to patch projection
    embeddings = self.spectrum_to_patch(inputs)

    # flatten inputs
    embeddings, input_shape = self._flatten_inputs(embeddings)

    if self.use_masking and training:
      # generate random masks and replace mask ids with special token mask_embd
      masked_embeddings, random_mask = self.random_embd_mask(embeddings)
    else:
      masked_embeddings = embeddings
      random_mask = tf.ones((get_shape(embeddings)[0:2]), dtype=tf.float32)

    # apply pre-tx projection - if applies
    masked_embeddings = self.pre_proj(masked_embeddings)

    # add modality-specific positional encoding embeddings
    masked_embeddings = self.pos_embedding_lookup(
        masked_embeddings,
        input_shape,
        training
        )

    # append special tokens: [agg]
    tx_inputs = self._append_special_tokens(masked_embeddings)

    # call Transformer
    outputs = self.tx(inputs=tx_inputs,
                      attention_mask=None,
                      training=training)

    # get last hidden states and perform final linear projection
    last_hidden_states = outputs["hidden_states"][-1]
    last_hidden_states = self.post_proj(last_hidden_states)
    output_shape = input_shape[:-1] + [get_shape(last_hidden_states)[-1]]

    aggregated = last_hidden_states[:, 0, :]
    predictions = last_hidden_states[:, 1:, :]
    predictions_2d = tf.reshape(predictions, output_shape)

    # add token-level Transformer outputs
    outputs["embeddings"] = embeddings
    outputs["random_mask"] = random_mask
    outputs["predictions"] = predictions
    outputs["predictions_2d"] = predictions_2d

    return aggregated, outputs
