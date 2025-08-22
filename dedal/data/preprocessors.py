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

"""Custom `seqio`-based preprocessors."""

from typing import Mapping, MutableMapping, Optional, Sequence, Union

import seqio
from t5.data import preprocessors as t5_preprocessors
import tensorflow as tf
import tensorflow_text as tf_text
import typing_extensions

# Type aliases
PRNGSeeds = tf.Tensor  # `tf.Tensor<int>[N, 2]`, with `N >= 1`.


@seqio.map_over_dataset
def cast_from_string(
    example,
    key_to_dtype_map,
):
  """Casts tf.Tensor<tf.string> elements to bool / numeric dtypes."""
  for key, dtype in key_to_dtype_map.items():
    if dtype == tf.bool:
      example[key] = tf.convert_to_tensor(example[key] == 'True', dtype=tf.bool)
    else:
      example[key] = tf.strings.to_number(example[key], dtype)
  return example


def filter_by_bool_key(
    dataset,
    key,
    negate_predicate = False,
):
  """Filters examples according to the value of `bool` field `example[key]`."""
  return dataset.filter(
      lambda ex: tf.logical_not(ex[key]) if negate_predicate else ex[key])


@seqio.map_over_dataset(num_seeds=1)
def random_crop(
    example,
    seed,
    sequence_length = None,
    max_length = None,
    add_eos = False,
    passthrough_feature_keys = None,
    feature_key = 'inputs',
):
  """Takes a random chunk out of each feature the size of `sequence_length`."""
  # TODO(fllinares): improve docstring.
  if passthrough_feature_keys and feature_key in passthrough_feature_keys:
    raise ValueError(f'passthrough_feature_keys cannot contain {feature_key}.')
  # `max_length` takes precedence over `sequence_length` to allow for extra
  # flexibility.
  if max_length is None and sequence_length is not None:
    max_length = sequence_length[feature_key]
  if max_length is None:
    raise ValueError('Either max_length or sequence_length[feature_key] must be'
                     ' set but both were None.')
  # If EOS token will be appended at a later step in the pipeline, we account
  # for it by reducing the effective maximum length by one.
  if add_eos:
    max_length -= 1

  tokens = example[feature_key]
  n_tokens = tf.size(tokens)
  start_idx = tf.random.stateless_uniform(
      shape=(),
      seed=seed,
      minval=0,
      maxval=tf.maximum(n_tokens - max_length + 1, 1),  # [min_val, max_val).
      dtype=tf.int32)
  cropped_tokens = tokens[start_idx:start_idx + max_length]

  return {
      feature_key: cropped_tokens,
      **{
          k: example[k]
          for k in example
          if passthrough_feature_keys and k in passthrough_feature_keys
      },
  }


def _random_where(
    x,
    y,
    prob_condition,
    seeds,
):
  """Randomly multiplexes two `tf.Tensor`s `x` and `y`."""
  values = tf.random.stateless_uniform(tf.shape(x), seed=seeds[0])
  condition = values < prob_condition
  return tf.where(condition, x, y)


class NoiseMaskFn(typing_extensions.Protocol):
  """Signature for functions returning a `seqio.preprocessors` noise mask."""

  def __call__(
      self,
      length,
      noise_density,
      seeds,
  ):
    """Returns a boolean mask of shape `[length]` marking some tokens as noisy.

    Args:
      length: The number of tokens to choose from.
      noise_density: The proportion of tokens that should be selected.
      seeds: a `tf.Tensor<int>[N, 2]` with the PRNG seeds for stateless sampling
        ops in `tf.random`. Some of these seeds may be ignored by the function.

    Returns:
      A `tf.Tensor<bool>[length]` with entries being `True` representing the
      subset of tokens chosen to be "noisy".
    """


def choose_k_noise_mask(
    length,
    noise_density,
    seeds,
):
  """Uniformly chooses k out of `length` tokens without replacement.

  Args:
    length: The number of tokens to choose from.
    noise_density: The proportion of tokens that should be selected. The number
      of chosen tokens `k` is given by `floor(noise_density * length)`
    seeds: a `tf.Tensor<int>[N, 2]` with the PRNG seed for stateless sampling
      ops in `tf.random`. If `N > 1`, the last `N - 1` rows of `seeds` will be
      ignored.

  Returns:
    A `tf.Tensor<bool>[length]` with exactly `k = floor(noise_density * length)`
    entries being `True` representing the subset of tokens chosen.
  """
  if not 0.0 <= noise_density <= 1.0:
    raise ValueError(
        f'noise density must be in [0, 1] but got {noise_density} instead.')
  # The number of noisy tokens `k` is deterministically computed as
  # `floor(noise_density * length)`.
  noise_density = tf.convert_to_tensor(noise_density, tf.float32)
  k = tf.cast(noise_density * tf.cast(length, tf.float32), tf.int32)

  # Sample `k` out of `length` indices uniformly at random without replacement,
  # using `tf.random` stateless ops for determinism.
  values = tf.random.stateless_uniform([length], seed=seeds[0])
  _, indices = tf.math.top_k(values, k=k, sorted=False)

  return tf.scatter_nd(
      indices=tf.expand_dims(indices, axis=-1),
      updates=tf.ones_like(indices, dtype=tf.bool),
      shape=[length])


def random_tokens_like(
    tokens,
    seeds,
    vocabulary,
    num_reserved_tokens = 3,
    exclude_extra_ids = True,
):
  """Creates a `tf.Tensor` like `tokens` with random tokens from `vocabulary`.

  Adapted from `t5.data.preprocessors.noise_token_to_random_token` by adding
  extra flexibility for excluding tokens from the vocabulary.

  Args:
    tokens: A `tf.Tensor<dtype>[shape]` whose dtype and shape determines those
      of the output.
    seeds: a `tf.Tensor<int>[N, 2]` with the PRNG seed for stateless sampling
      ops in `tf.random`. If `N > 1`, the last `N - 1` rows of `seeds` will be
      ignored.
    vocabulary: the `seqio.Vocabulary` from which tokens should be sampled.
    num_reserved_tokens: the number of special control tokens in the vocabulary
      which should *not* be sampled. It is assumed that these *always* are the
      first tokens in the vocabulary, i.e., the excluded tokens will be in the
      range `[0, num_reserved_tokens)`.
    exclude_extra_ids: Whether to prevent any extra ID tokens from being sampled
      from the vocabulary.

  Returns:
    A `tf.Tensor` with the same `shape` and `dtype` as `tokens` but populated
    with random tokens uniformly sampled i.i.d. from `vocabulary`.
  """
  effective_vocab_size = vocabulary.vocab_size
  if exclude_extra_ids:
    effective_vocab_size -= vocabulary.extra_ids
  return tf.random.stateless_uniform(
      tf.shape(tokens),
      minval=num_reserved_tokens,
      maxval=effective_vocab_size,
      dtype=tokens.dtype,
      seed=seeds[0])


def bert_denoising(
    dataset,
    output_features,
    noise_density = 0.15,
    mask_prob = 0.8,
    replace_prob = 0.10,
    noise_mask_fn = choose_k_noise_mask,
    exclude_reserved_tokens = True,
    num_reserved_tokens = 3,
    exclude_extra_ids = True,
    replace_vocabulary = None,
    passthrough_feature_keys = None,
    input_feature_key = 'inputs',
    target_feature_key = 'targets',
    noise_mask_feature_key = 'noise_mask',
    **unused_kwargs,
):
  """Creates a masked language modelling task, following BERT."""
  if passthrough_feature_keys:
    for key in (input_feature_key, target_feature_key, noise_mask_feature_key):
      if key in passthrough_feature_keys:
        raise ValueError(f'passthrough_feature_keys cannot contain {key}.')

  if mask_prob < 0.0:
    raise ValueError(
        f'mask_prob must be non-negative but got {mask_prob} instead.')
  if replace_prob < 0.0:
    raise ValueError(
        f'replace_prob must be non-negative but got {replace_prob} instead.')
  corrupt_prob = mask_prob + replace_prob
  if not 0.0 < corrupt_prob < 1.0:
    raise ValueError(
        f'mask_prob + replace_prob must be in (0, 1) {corrupt_prob} instead.')
  # Since `corrupt_prob > 0`, `replace_ratio` is always well-defined.
  replace_ratio = replace_prob / corrupt_prob

  vocabulary = output_features[target_feature_key].vocabulary
  if (input_feature_key in output_features and
      vocabulary != output_features[input_feature_key].vocabulary):
    raise ValueError(
        f'bert_denoising creates {input_feature_key} based on tokenized '
        f'{target_feature_key} but was applied to a task that uses different '
        f'vocabularies for {input_feature_key} and {target_feature_key}.')
  effective_vocab_size = vocabulary.vocab_size
  if exclude_extra_ids:
    effective_vocab_size -= vocabulary.extra_ids
  sentinel_id = t5_preprocessors.sentinel_id(vocabulary)
  # If no `replace_vocabulary` is specified, default to `vocabulary`.
  replace_vocabulary = replace_vocabulary or vocabulary

  @seqio.map_over_dataset(num_seeds=5)
  def bert_denoising_fn(
      example,
      seeds,
  ):
    """Instantiates BERT-like denoising task for one input example."""
    tokens = example[target_feature_key]

    # We pass two seeds to `noise_mask_fn` for broad compatibility with other
    # functions in `t5_preprocessors`.
    noise_mask = noise_mask_fn(tf.size(tokens), noise_density, seeds=seeds[:2])
    if exclude_reserved_tokens:
      noise_mask = tf.logical_and(noise_mask, tokens >= num_reserved_tokens)
    if exclude_extra_ids:
      noise_mask = tf.logical_and(noise_mask, tokens < effective_vocab_size)

    # BERT-style [MASK] tokens.
    sentinel_tokens = tf.fill(
        dims=tf.shape(tokens),
        value=tf.cast(sentinel_id, tokens.dtype))
    # `random_tokens` are sampled uniformly at random with replacement from
    # `replace_vocabulary`.
    random_tokens = random_tokens_like(
        tokens=tokens,
        seeds=tf.expand_dims(seeds[2], axis=0),
        vocabulary=replace_vocabulary)
    # Each entry in `corrupted_tokens` is randomly multiplexed i.i.d. such that
    #  it comes from either
    # + `sentinel_tokens` with probability `mask_prob`,
    # + `random_tokens` with probability `replace_prob`,
    # + `tokens` with probability  `1.0 - (mask_prob + replace_prob)`.
    corrupted_tokens = _random_where(
        _random_where(
            random_tokens,
            sentinel_tokens,
            prob_condition=replace_ratio,
            seeds=tf.expand_dims(seeds[3], axis=0)),
        tokens,
        prob_condition=corrupt_prob,
        seeds=tf.expand_dims(seeds[4], axis=0))

    # Arbitrarily encodes `True` as 1 and `False` as 2, reserving 0 for padding.
    # Otherwise, using 0 for `False` appears to confuse the sequence packing ops
    # downstream in the data pipeline.
    encoded_noise_mask = tf.cast(tf.where(noise_mask, 1, 2), tf.int32)

    return {
        input_feature_key: tf.where(noise_mask, corrupted_tokens, tokens),
        target_feature_key: tokens,
        noise_mask_feature_key: encoded_noise_mask,
        **{
            k: example[k]
            for k in example
            if passthrough_feature_keys and k in passthrough_feature_keys
        },
    }

  # We follow `seqio` and disable `no-value-for-parameter`` to prevent false
  # positives when seeds are injected by `seqio.map_over_dataset`.
  return bert_denoising_fn(dataset)  # pylint: disable=no-value-for-parameter


@seqio.map_over_dataset
def maybe_append_eos(
    example,
    output_features,
    sequence_key = 'sequence',
    flag_key = 'eos',
):
  """Extends SeqIO's `append_eos` to allow for per-example."""
  if sequence_key in output_features and output_features[sequence_key].add_eos:

    def _with_eos():
      value = example[sequence_key]
      eos_id = output_features[sequence_key].vocabulary.eos_id
      return seqio.preprocessors._append_to_innermost_axis(value, eos_id)  # pylint: disable=protected-access

    example[sequence_key] = tf.cond(
        pred=example[flag_key],
        true_fn=_with_eos,
        false_fn=lambda: example[sequence_key])
  return example


@seqio.map_over_dataset
def uncompress_states(
    example,
    states_key = 'states',
):
  r"""Uncompresses a `tf.Tensor<tf.string>` representing an alignment path.

  Alignment path strings are compressed by substituting consecutive stretches of
  identical characters by the number of occurrences followed by a single
  instance of the character. For example, `SMMMMMXXXXMM` would be compressed and
  stored as as `1S5M4X2M`. This function inverts this compression mapping.

  Args:
    example: An example of the input pipeline, represented as a collection of
      `tf.Tensor`-valued fields indexed by string keys.
    states_key: The name of the field holding the compressed alignment path
      string. It is assumed that `example[states_key]` conforms to the regex
      `r'^(\d+[SMXY])*$'`.

  Returns:
    The input `example` with all fields unchanged except `example[states_key]`,
    which is replaced by the corresponding uncompressed alignment path string.
  """
  states = tf_text.regex_split(example[states_key], r'(\d+)')[0]
  counts = tf.strings.to_number(
      tf_text.regex_split(example[states_key], r'([SMXY])')[0],
      out_type=tf.int32)
  num_chunks = tf.shape(states)[0]
  chunks = tf.TensorArray(dtype=tf.string, size=num_chunks)
  for n in tf.range(num_chunks):
    chunk = tf.strings.reduce_join(tf.repeat(states[n], counts[n]))
    chunks = chunks.write(n, chunk)
  example[states_key] = tf.strings.reduce_join(chunks.stack())
  return example
