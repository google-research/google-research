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

"""Preprocessors for Fast Decoding project."""

import functools
from typing import Any, Mapping, Optional

import seqio
from t5.data import preprocessors as t5_preprocessors
import tensorflow.compat.v2 as tf

from padir.padir import config_options
from padir.padir.utils import vocab_utils


# T5 pretraining objective.
SPAN_CORRUPTION = [
    functools.partial(
        t5_preprocessors.rekey, key_map={'inputs': None, 'targets': 'text'}
    ),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    functools.partial(
        t5_preprocessors.span_corruption,
        merge_examples_to_reduce_padding=False,
        mean_noise_span_length=3,
        noise_density=0.15,
    ),
    seqio.preprocessors.append_eos_after_trim,
]


# T5 pretraining objective, but with much longer (sentence-size) noise spans.
SPAN_CORRUPTION_LONG = [
    functools.partial(
        t5_preprocessors.rekey, key_map={'inputs': None, 'targets': 'text'}
    ),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    functools.partial(
        t5_preprocessors.span_corruption,
        merge_examples_to_reduce_padding=False,
        mean_noise_span_length=32,
        noise_density=0.5,
    ),
    seqio.preprocessors.append_eos_after_trim,
]


# Prefix language modeling objective.
PREFIX_LM = [
    functools.partial(
        t5_preprocessors.rekey, key_map={'inputs': None, 'targets': 'text'}
    ),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    t5_preprocessors.prefix_lm,
    seqio.preprocessors.append_eos_after_trim,
]


def create_range_mask(first_mask_id, mask_length):
  """Returns a mask where each position uses a different mask id."""
  return tf.range(first_mask_id, first_mask_id + mask_length, dtype=tf.int32)


def token_id_position(
    tensor,
    dtype = tf.int32,
    token_id = 0,
):
  """Return a tensor with 1 in positions where id occurs."""
  return tf.cast(tf.equal(tensor, token_id), dtype=dtype)


def combine_masks(
    masks,
    dtype = tf.int32,
):
  """Combine binary masks (logical or)."""
  assert len(masks) >= 1
  combined_mask = tf.cast(masks[0], dtype=tf.bool)
  for mask in masks[1:]:
    combined_mask = tf.logical_or(combined_mask, tf.cast(mask, dtype=tf.bool))
  return tf.cast(combined_mask, dtype=dtype)


def noise_token_to_correct_token_or_random_token_or_sentinel(
    tokens,
    noise_mask,
    vocabulary,
    seeds,
    correct_prob = 0.1,
    random_prob = 0.1,
):
  """Replace each noise token with the correct token, a random token, or a sentinel.

  For each masked token:
  1. With probability correct_prob, we replace it by the correct/original token.
  2. With probability random_prob, we replace it by a
  random token from the vocabulary.
  3. Otherwise, we replace it with a sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    vocabulary: a vocabulary.Vocabulary
    seeds: an int32 Tensor, shaped (2, 2).
    correct_prob: a float
    random_prob: a float

  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  if correct_prob is not None:
    assert 0 <= correct_prob < 1
    if random_prob is not None:
      assert 0 <= correct_prob + random_prob <= 1
  if random_prob is not None:
    assert 0 <= random_prob <= 1

  seeds = tf.unstack(tf.random.experimental.stateless_split(seeds[0], 5))
  if correct_prob is None:
    correct_prob = tf.random.stateless_uniform([], seeds[0], maxval=0.5)
  if random_prob is None:
    random_prob = tf.random.stateless_uniform([], seeds[1], maxval=0.5)
  use_correct = (
      tf.random.stateless_uniform(tf.shape(tokens), seed=seeds[2])
      < correct_prob
  )
  return tf.where(
      use_correct,
      tokens,
      t5_preprocessors.noise_token_to_random_token_or_sentinel(
          tokens,
          noise_mask,
          vocabulary,
          seeds=seeds[-2:],
          random_prob=random_prob / (1 - correct_prob),
      ),
  )


def single_example_target_denoise(
    features,
    seed,
    *,
    output_features,
    noise_density,
    noise_mask_fn,
    targets_fn,
):
  """Single example target denoising."""
  seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 4))
  targets = features['targets']
  vocabulary = output_features['targets'].vocabulary
  if noise_density is None:
    noise_density = tf.random.stateless_uniform(
        [],
        seed,
        minval=0.0,
        maxval=1.0,
        dtype=tf.dtypes.float32,
    )
  noise_mask = noise_mask_fn(tf.size(targets), noise_density, seeds=seeds[:2])
  targets_masked = targets_fn(targets, noise_mask, vocabulary, seeds=seeds[2:])
  return {
      'targets_masked': targets_masked,
      'noise_mask': tf.cast(noise_mask, tf.int32),
      **{k: features[k] for k in features},
  }


def padir_denoising(
    dataset,
    noise_density,
    correct_prob,
    random_prob,
    use_range_mask,
    max_target_len,
    output_features,
):
  """Adds iid noise to the target for a denoising training task."""
  targets_fn = functools.partial(
      noise_token_to_correct_token_or_random_token_or_sentinel,
      correct_prob=correct_prob,
      random_prob=random_prob,
  )

  @seqio.map_over_dataset(num_seeds=1)
  def my_fn(features, seed):
    features = single_example_target_denoise(
        features,
        seed,
        output_features=output_features,
        noise_density=noise_density,
        noise_mask_fn=t5_preprocessors.iid_noise_mask,
        targets_fn=targets_fn,
    )
    if not use_range_mask:
      return features

    # single_example_target_denoise uses a single [MASK] token.
    # If during inference, the decoder is initialized with a range mask
    # (i.e., [MASK_0], [MASK_1], ..., [MASK_N]), then we need to
    # replace each [MASK] with its corresponding [MASK_i] where i denotes the
    # position in the sequence.
    features = dict(features)
    targets_masked = features['targets_masked']
    cur_target_len = tf.size(targets_masked)
    # NB: targets_masked may be shorter or longer than max_target_len.
    # So we create a new range mask instead of relying on initial_infer_mask.
    # We do not truncate targets_masked here as truncating is done at the end of
    # preprocessing.
    output_vocab = output_features['targets'].vocabulary
    mask_id = vocab_utils.get_mask_id(output_vocab)
    start_id, _ = vocab_utils.get_mask_id_range(output_vocab, max_target_len)
    infer_mask = create_range_mask(start_id, cur_target_len)
    features['targets_masked'] = tf.where(
        targets_masked >= mask_id,
        infer_mask,
        targets_masked,
    )
    return features

  return my_fn(dataset)  # pylint: disable=no-value-for-parameter


@seqio.map_over_dataset
def trim_non_eos_features(
    feature_dict,
    max_target_length,
):
  """Trims decoder related features that do not end in EOS."""
  out = dict(feature_dict)
  noise_mask = feature_dict['noise_mask']
  noise_mask = noise_mask[: max_target_length - 1]  # Leave room for EOS
  noise_mask = tf.concat([noise_mask, [0]], axis=0)  # EOS is not a masked token
  out['noise_mask'] = noise_mask
  return out


@seqio.map_over_dataset
def add_loss_weights(
    feature_dict,
    weight_scheme,
    output_features,
):
  """Adds the 'decoder_loss_weights' feature based on some weighting scheme."""
  out = dict(feature_dict)

  targets_masked = feature_dict['targets_masked']  # possibly with bos and eos
  bos_id = vocab_utils.get_bos_id(output_features['targets'].vocabulary)
  mask_id = vocab_utils.get_mask_id(output_features['targets'].vocabulary)
  eos_id = output_features['targets'].vocabulary.eos_id
  bos_and_eos_mask = combine_masks([
      token_id_position(targets_masked, tf.int32, bos_id),
      token_id_position(targets_masked, tf.int32, eos_id),
  ])

  if weight_scheme == config_options.LossWeightingScheme.ALL:
    weights = tf.ones_like(feature_dict['targets'])
  elif weight_scheme == config_options.LossWeightingScheme.NOISE:
    noise_mask = feature_dict['noise_mask']
    weights = combine_masks([noise_mask, bos_and_eos_mask])
  else:  # weight_scheme == config_options.LossWeightingScheme.MASK:
    weights = combine_masks(
        [tf.greater_equal(targets_masked, mask_id), bos_and_eos_mask]
    )
  out['decoder_loss_weights'] = weights
  return out
