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

"""Core code of VCT: the temporal entropy model."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
from typing import NamedTuple, Optional

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

from vct.src import auxiliary_layers
from vct.src import bottlenecks
from vct.src import metric_collection
from vct.src import patcher
from vct.src import transformer_layers


_LATENT_NORM_FAC: float = 35.0


def _unbatch(t, dims):
  """Reshapes first dimension, i.e. (b, ...) becomes (b', *dims, ...)."""
  b_in, *other_dims = t.shape
  b_out = b_in // np.prod(dims)
  return tf.reshape(t, (b_out, *dims, *other_dims))


class TemporalEntropyModelOut(NamedTuple):
  """Output of a temporal entropy model.

  Attributes:
    perturbed_latent: Quantized/noised latent.
    bits: Bits taken to transmit the latent. Shape: (b,).
    metrics: Metrics collected by the entropy model. Shape: (b,).
    features: Features of the entropy model to be used by a synthesis
      transform for dequantizing. Shape: (b, h, w, d_model).
  """

  perturbed_latent: tf.Tensor
  bits: tf.Tensor
  metrics: metric_collection.Metrics
  features: Optional[tf.Tensor] = None


class PreviousLatent(NamedTuple):
  """Represents a single previous latent.

  Attributs:
    quantized: The quantized latent.
    processed: We process the quantized
      latent with the encoder. See `process_previous_latent_q` for info.
  """

  quantized: tf.Tensor
  processed: tf.Tensor


class VCTEntropyModel(tf.Module):
  """Temporal Entropy model."""

  def __init__(
      self,
      num_channels,
      context_len = 2,
      window_size_enc = 8,
      window_size_dec = 4,
      num_layers_encoder_sep = 3,
      num_layers_encoder_joint = 2,
      num_layers_decoder = 5,
      d_model = 768,
      num_head = 16,
      mlp_expansion = 4,
      drop_out_enc = 0.0,
      drop_out_dec = 0.0,
  ):
    """Initializes model.

    Args:
      num_channels: Number of channels of the latents, i.e., symbols per token.
      context_len: How many previous latents to expect.
      window_size_enc: Window size in encoder.
      window_size_dec: Window size in decoder.
      num_layers_encoder_sep: Sepearte layer count.
      num_layers_encoder_joint: Joint layer count.
      num_layers_decoder: Number of decoder layers.
      d_model: Feature dimensionality inside the model.
      num_head: Number of attention heads per Multi-Head Attention layer.
      mlp_expansion: Expansion factor in feature dimensionality for each MLP.
      drop_out_enc: Sets the drop_out probability for various places in the
        encoder.
      drop_out_dec: Sets the drop_out probability for various places in the
        decoder.
    """
    if window_size_enc < window_size_dec:
      raise ValueError("Invalid config.")
    if num_channels < 0:
      raise ValueError("Invalid config.")
    super().__init__()
    self.num_channels = num_channels
    self.d_model = d_model
    self.bottleneck = bottlenecks.ConditionalLocScaleShiftBottleneck(
        round_indices=False, coding_rank=0)
    self.range_bottleneck = None

    self.context_len = context_len
    self.encoder_sep = transformer_layers.EncoderSection(
        drop_out=drop_out_enc,
        num_layers=num_layers_encoder_sep,
        name="encoder_sep",
        d_model=d_model,
        num_head=num_head,
        mlp_expansion=mlp_expansion)
    self.encoder_joint = transformer_layers.EncoderSection(
        drop_out=drop_out_enc,
        num_layers=num_layers_encoder_joint,
        name="encoder_joint",
        d_model=d_model,
        num_head=num_head,
        mlp_expansion=mlp_expansion)

    self.patcher = patcher.Patcher(window_size_dec, "REFLECT")
    self.learned_zero = auxiliary_layers.StartSym(num_channels)

    self.window_size_enc = window_size_enc
    self.window_size_dec = window_size_dec
    self.enc_position_sep = auxiliary_layers.LearnedPosition(
        "enc_sep", window_size_enc**2, d_model)
    self.enc_position_joint = auxiliary_layers.LearnedPosition(
        "enc_joint", window_size_enc**2 * context_len, d_model)
    self.dec_position = auxiliary_layers.LearnedPosition(
        "dec", window_size_dec**2, d_model)

    self.seq_len_dec = (window_size_dec ** 2)

    self.post_embedding_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-5, name="after_embedding")

    self.encoder_embedding = auxiliary_layers.make_embedding_layer(
        num_channels, d_model)
    self.decoder_embedding = auxiliary_layers.make_embedding_layer(
        num_channels, d_model)

    self.decoder = transformer_layers.Transformer(
        is_decoder=True,
        num_layers=num_layers_decoder,
        d_model=d_model,
        seq_len=self.seq_len_dec,
        num_head=num_head,
        mlp_expansion=mlp_expansion,
        drop_out=drop_out_dec,
    )

    def _make_final(output_channels):
      return tf.keras.Sequential([
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dense(output_channels),
      ])

    self.mean = _make_final(num_channels)
    self.scale = _make_final(num_channels)

  def prepare_for_range_coding(self):
    if self.range_bottleneck is None:
      logging.info("Preparing model for range coding...")
      # Takes ~1min for 100 means and 64 scales.
      self.range_bottleneck = (
          bottlenecks.ConditionalLocScaleShiftBottleneck(
              coding_rank=2, compression=True, round_indices=True,
              num_means=100, num_scales=256))

  def process_previous_latent_q(self, previous_latent_quantized,
                                training):
    """Processes the previous via the encoder, see baseclass docstring.

    This can be used if previous latents go through an expensive transform
    before being fed to the entropy model, and will be stored in the `processed`
    field of the `PreviousLatent` tuple.

    The output of this function applied to all quantized latents should
    be fed to `__call__`. Note that this is used to improve efficiency,
    as it avoids calling expensive processing of previous latents at
    each time step. For an example, see beachcomber/models.py.

    Args:
      previous_latent_quantized: The previous latent to process.
      training: Whether we are training.

    Returns:
      Processed previous latent as a `PreviousLatent` tuple.
    """
    # (b', seq_len, num_channels)
    latent_patches, _ = self.patcher(
        previous_latent_quantized, self.window_size_enc)
    patches = latent_patches / _LATENT_NORM_FAC

    # (b', seq_len, d_model)
    patches = self.encoder_embedding(patches)
    patches = self.post_embedding_norm(patches)
    patches = self.enc_position_sep(patches)
    patches = self.encoder_sep(patches, training)

    return PreviousLatent(previous_latent_quantized, processed=patches)

  def _embed_latent_q_patched(self, latent_q_patched):
    """Embeds current latent for decoder."""
    # (b', seq_len, num_channels)
    latent_q_patched = latent_q_patched / _LATENT_NORM_FAC
    # (b', seq_len, d_model)
    latent_q_patched = self.decoder_embedding(latent_q_patched)
    latent_q_patched = self.post_embedding_norm(latent_q_patched)
    return self.dec_position(latent_q_patched)

  def _get_transformer_output(
      self,
      *,
      encoded_patched,
      latent_q_patched,
      training,
      true_batch_size,
  ):
    """Calculates transformer distribution prediction."""
    if encoded_patched.shape[-1] != self.d_model:
      raise ValueError(f"Context must have final dim {self.d_model}, "
                       f"got shape={encoded_patched.shape}. "
                       "Did you run `process_previous_latent_q`?")

    latent_q_patched_shifted = self.learned_zero(latent_q_patched)
    latent_q_patched_emb_shifted = self._embed_latent_q_patched(
        latent_q_patched_shifted)
    del latent_q_patched  # Should not use after this line.

    tf.debugging.assert_shapes([
        (encoded_patched, ("B", "seq_len_enc", "d_model")),
        (latent_q_patched_emb_shifted, ("B", "seq_len_dec", "d_model")),
    ])

    encoded_patched = self.enc_position_joint(encoded_patched)
    encoded_patched = self.encoder_joint(encoded_patched, training)

    dec_output = self.decoder(
        latent=latent_q_patched_emb_shifted,
        enc_output=encoded_patched,
        training=training)

    mean = self.mean(dec_output)
    scale = self.scale(dec_output)

    return mean, scale, dec_output, metric_collection.Metrics.make()

  def _get_encoded_seqs(
      self,
      previous_latents,
      latent_shape
  ):
    encoded_seqs = [p.processed for p in previous_latents]
    if len(encoded_seqs) < self.context_len:
      if self.context_len == 2:
        return [encoded_seqs[0], encoded_seqs[0]]
      if self.context_len == 3:
        if len(encoded_seqs) == 1:
          return [encoded_seqs[0], encoded_seqs[0], encoded_seqs[0]]
        else:
          assert len(encoded_seqs) == 2  # Sanity.
          return [encoded_seqs[0], encoded_seqs[0], encoded_seqs[1]]
      raise ValueError(f"Unsupported: {self.context_len}")
    return encoded_seqs

  def __call__(
      self,
      latent_unquantized,
      previous_latents,
      training,
  ):
    """Does a forward pass through the entropy model.

    Args:
      latent_unquantized: Latent to transmit.
      previous_latents: Previous latents.
      training: Whether we are training.

    Returns:
      TemporalEntropyModelOut, see docstring there.
    """

    b, h, w, _ = latent_unquantized.shape
    encoded_seqs = self._get_encoded_seqs(previous_latents, (h, w))
    b_enc, _, d_enc = encoded_seqs[0].shape
    if d_enc != self.d_model:
      raise ValueError(encoded_seqs[0].shape)

    latent_q = tfc.round_st(latent_unquantized)

    latent_q_patched, (n_h, n_w) = self.patcher(latent_q, self.window_size_dec)
    b_dec, _, d_dec = latent_q_patched.shape
    if d_dec != self.num_channels:
      raise ValueError(latent_q_patched.shape)
    if b_dec != b_enc:
      raise ValueError(
          f"Expected matching batch dimes, got {b_enc} != {b_dec}!")

    mean, scale, dec_output, metrics = self._get_transformer_output(
        # Fuse all in the sequence dimension.
        encoded_patched=tf.concat(encoded_seqs, axis=-2),
        latent_q_patched=latent_q_patched,
        training=training,
        true_batch_size=b)
    assert mean.shape == latent_q_patched.shape
    decoder_features = self.patcher.unpatch(dec_output, n_h, n_w, crop=(h, w))

    # Each tensor here is (b', seq_len, num_channels).
    latent_unquantized_patched = self.patcher(latent_unquantized,
                                              self.window_size_dec).tensor
    output, bits = self.bottleneck(
        latent_unquantized_patched,
        mean,
        scale,
        training=training)
    assert output.shape == bits.shape

    # (b, h, w, num_channels).
    output = self.patcher.unpatch(output, n_h, n_w, crop=(h, w))
    assert output.shape == latent_unquantized.shape

    # (b,)
    bits_per_batch = tf.reduce_sum(_unbatch(bits, (n_h, n_w)), (1, 2, 3, 4))
    assert bits_per_batch.shape == [latent_unquantized.shape[0]]

    metrics.record_scalar("bits/total", tf.reduce_mean(bits_per_batch))
    metrics.record_scalar("mean_quantization_error",
                          tf.reduce_mean(output - latent_unquantized))

    return TemporalEntropyModelOut(
        output,
        bits=bits_per_batch,
        metrics=metrics,
        features=decoder_features)

  @tf.function(jit_compile=True, autograph=False)
  def _get_mean_scale_jitted(
      self,
      *,
      encoded_patched,
      latent_q_patched,
      true_batch_size,
  ):
    """Jitted version of `_get_transformer_output`, for `validate_causal`."""
    mean, scale, dec_output, _ = self._get_transformer_output(
        encoded_patched=encoded_patched,
        latent_q_patched=latent_q_patched,
        training=False,
        true_batch_size=true_batch_size,
        )
    return mean, scale, dec_output

  def validate_causal(self, latent,
                      previous_latents):
    """Validates that the model is causal."""
    if self.range_bottleneck is not None:
      logging.warning("Not validating causality due to range coding...")
      return

    if not tf.executing_eagerly():
      logging.warning("Not validating causality in graph mode...")
      return

    h, w = self.window_size_dec, self.window_size_dec  # We use a single patch.
    encoded_seqs = self._get_encoded_seqs([previous_latents[-1]], (h, w))
    encoded_patched = tf.concat([p[:1, Ellipsis] for p in encoded_seqs], -2)
    latent_q_patched, _ = self.patcher(latent, self.window_size_dec)
    latent_q_patched = latent_q_patched[:1, Ellipsis].numpy()
    # First we run the model as we usually do, relying on masks to guarantee
    # causality.
    masked_means, masked_scales, _ = self._get_mean_scale_jitted(
        encoded_patched=encoded_patched,
        latent_q_patched=latent_q_patched,
        true_batch_size=1,
        )

    # Then we run the model iteratively, feeding progressivly more input.
    current_inp = np.full_like(latent_q_patched, fill_value=10.)
    autoreg_means = np.full_like(latent_q_patched, fill_value=10.)
    autoreg_scales = np.full_like(latent_q_patched, fill_value=10.)

    for i in range(self.seq_len_dec):
      # Note that the transformer starts using a zero sybmol, so internally,
      # it shifts the input to the right. Thus, on the very first call (where
      # i == 0), we want a dummy input. Only afterwards we start filling in
      # input, using it shifted.
      if i > 0:
        current_inp[:, i - 1, :] = latent_q_patched[:, i - 1, :]
      mean_i, scale_i, _ = self._get_mean_scale_jitted(
          encoded_patched=encoded_patched,
          latent_q_patched=current_inp,
          true_batch_size=1)
      autoreg_means[:, i, :] = mean_i[:, i, :]
      autoreg_scales[:, i, :] = scale_i[:, i, :]

    err = (autoreg_means - masked_means)[0, Ellipsis]
    tf.debugging.assert_near(autoreg_means, masked_means, summarize=10,
                             message=str(err))
    tf.debugging.assert_near(autoreg_scales, masked_scales, summarize=10)

  ##############################################################################
  # Range Coding Helpers #######################################################
  ##############################################################################

  def range_code(
      self,
      *,
      latent_unquantized,
      previous_latents,
      run_decode,
  ):
    """Like `__call__` but uses (autoregressive) range coding."""
    assert self.range_bottleneck is not None

    b, h, w, c = latent_unquantized.shape
    encoded_seqs = self._get_encoded_seqs(previous_latents, (h, w))
    encoded = tf.concat(encoded_seqs, -2)
    assert b == 1
    latent_patched, (n_h, n_w) = self.patcher(latent_unquantized,
                                              self.window_size_dec)

    # Encoding.
    bytestrings, means, scales, dec_output = self._encode(
        latent_patched, encoded)
    decoder_features = self.patcher.unpatch(dec_output, n_h, n_w, crop=(h, w))

    # Count bits
    bits = tf.constant(
        [sum(len(bytestring.numpy()) * 8 for bytestring in bytestrings)],
        dtype=tf.float32)

    # Decoding.
    if not run_decode:
      # For performance, we skip the real decode after frame 5. However,
      # we still use range coded bitrates, and we assert that we get the
      # same result for the first 5.
      logging.log_first_n(logging.WARNING, "Skipping real decode...", n=20)
      latent_q = tf.round(latent_unquantized)
    else:
      latent_q = self._decode(bytestrings, encoded, shape=(h, w, c),
                              encode_means=means, encode_scales=scales)

    return TemporalEntropyModelOut(
        latent_q,
        bits,
        metrics=metric_collection.Metrics.make(),
        features=decoder_features)

  def _encode(self, latent_patched, encoded):
    bytestrings = []
    current_inp = np.full_like(latent_patched, fill_value=100.)
    autoreg_means = np.full_like(latent_patched, fill_value=100.)
    autoreg_scales = np.full_like(latent_patched, fill_value=100.)
    dec_output_shape = (*latent_patched.shape[:-1], self.d_model)
    dec_output = np.full(dec_output_shape, fill_value=100., dtype=np.float32)
    prev_mean = None
    prev_scale = None

    # We add 0 to code the very last symbol, it will become 0 - 1 == -1.
    for i in itertools.chain(range(self.seq_len_dec), [0]):
      # On the very first pass, we have no `prev_mean`, since we will feed
      # the zero symbol first to get an initial distribution.
      if prev_mean is not None:
        latent_i = latent_patched[:, i - 1, :]
        quantized_i, bytestring = self.range_bottleneck.compress(
            latent_i, prev_mean, prev_scale)
        assert bytestring.shape == ()  # pylint: disable=g-explicit-bool-comparison
        bytestrings.append(bytestring)
        current_inp[:, i - 1, :] = quantized_i
        if i == 0:
          break
      mean_i, scale_i, dec_output_i = self._get_mean_scale_jitted(
          encoded_patched=encoded,
          latent_q_patched=current_inp,
          true_batch_size=1)
      prev_mean = autoreg_means[:, i, :] = mean_i[:, i, :]
      prev_scale = autoreg_scales[:, i, :] = scale_i[:, i, :]
      dec_output[:, i, :] = dec_output_i[:, i, :]

    return bytestrings, autoreg_means, autoreg_scales, dec_output

  def _decode(self, bytestrings, encoded, shape,
              encode_means, encode_scales):
    h, w, c = shape
    fake_patched, (n_h, n_w) = self.patcher(
        tf.zeros((1, h, w, c)), self.window_size_dec)
    current_inp = np.full_like(fake_patched, fill_value=10.)
    prev_mean = None
    prev_scale = None
    for i in itertools.chain(range(self.seq_len_dec), [0]):
      if prev_mean is not None:
        decompressed_i = self.range_bottleneck.decompress(
            bytestrings.pop(0), prev_mean, prev_scale)
        current_inp[:, i - 1, :] = decompressed_i
        if i == 0:
          break
      mean_i, scale_i, _ = self._get_mean_scale_jitted(
          encoded_patched=encoded,
          latent_q_patched=current_inp,
          true_batch_size=1)
      # NOTE: We use the means from encoding, and log errors. Non-determinism
      # makes some outputs blow up ever so rarely (once every 100 or so
      # symbols).
      target_mean = encode_means[:, i, :]
      target_scale = encode_scales[:, i, :]
      actual_mean = mean_i[:, i, :]
      actual_scale = scale_i[:, i, :]

      error_mean = tf.reduce_sum(tf.abs(actual_mean - target_mean))
      error_scale = tf.reduce_sum(tf.abs(actual_scale - target_scale))

      logging.info("Decode step %i, mean err=%f, scale err=%f",
                   i, error_mean, error_scale)

      prev_mean = target_mean
      prev_scale = target_scale

    assert not bytestrings
    return self.patcher.unpatch(current_inp, n_h, n_w, crop=(h, w))
