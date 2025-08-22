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

"""Get predicted recommendations for data in a datasource.

The only function intednded for external calls is predict_from_datasource
"""

import collections
from collections.abc import Sequence, Generator
import functools
from typing import Union, Optional, TypeVar

from absl import logging
from flax import jax_utils
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from study_recommend import datasource as datasource_lib
from study_recommend import models
from study_recommend import types
from study_recommend.utils import input_pipeline_utils


FIELDS = types.ModelInputFields


def _infer_step(
    params,
    model_class,
    inputs,
    config,
    n_logits_to_discard,
    top_k = None,
):
  """Get the predicitions from given batch inputs.

  Args:
    params: flax.nn.Module parameter dict.
    model_class: The class of the model to use for inference.
    inputs: A single batch of inputs to do inference on.
    config: The configuration of the model to use for inference.
    n_logits_to_discard: The final n_logits_to_discard logits will be discarded
      before computing the top k recommendation. This allows us to enforce that
      the model does not recommend the separator_token or oov_token.
    top_k: If given, this function will return for each timepoint the batch the
      top k recommendations. If top_k=None the raw logits will be returned/

  Returns:
    An jax array (float) of logits if top_k is not None. Otherwise a jax
      array (int) of indices of top k recommendation items. When top_k is used
      we returnt the top scored items excluding the 0th logit and
      the last n_logits_to_discard logits.
  """
  logits = model_class(config).apply({'params': params}, inputs)
  if top_k is None:
    return logits

  # Discard logit for special tokens at the end as well as for 0 padding
  logits = logits[Ellipsis, 1:-n_logits_to_discard]

  _, top_k_indices = jax.lax.top_k(logits, top_k)
  # We need to add 1 to the indices as we removed the 0th logit
  return top_k_indices + 1

T = TypeVar('T')


def _per_student_iterator(
    separator_joint_list,
    list_to_split,
    separator,
):
  """Yields student sublists from a list separator-joint multi-student list.

  Assumes the structure of the list in list_to_split and in
  separator_joint_list is the same. It assumes separator_joint_list contains
  sequences of data corresponding to multiple students separated by entries
  equal to `separator`.

  For example if the list values were
  separator_joint_list  =   [ 1,    2, <separator>,    3, <separator>,   4,   5]
  list_to_split         =   ['a', 'b',         'c',  'd',         'e', 'f', 'g']
  Then this generator would yield the following lists.
  ['a', 'b']
  ['d']
  ['f', 'g']

  Args:
    separator_joint_list: The list with separator tokens. This is used to infer
      the structure of list_to_split with regard to student sublists.
    list_to_split: The list to yield sublits from.
    separator: The separator token value.

  Yields:
    sublists from list_to_split refering to single students. Separator tokens
    are removed from sublists before yielding.
  """
  assert len(separator_joint_list) == len(list_to_split)

  if not separator_joint_list:
    # No data passed. Terminate iterator.
    return

  # This assertion allows us to not handle edge case behaviour when the first
  # value is a separator token
  assert separator_joint_list[0] != separator

  start_index = 0
  length = len(separator_joint_list)
  for end_index in range(1, length + 1):
    if end_index == length or separator_joint_list[end_index] == separator:
      # This check stops us from yielding an empty list when their is a
      # terminal separator value or consecutive separator values.
      if start_index < end_index:
        yield list_to_split[start_index:end_index]
      # We set start_index to end_index + 1 to not include the separator token
      # in the yielded list.
      start_index = end_index + 1


def _batched_per_student_iterator(
    batched_separator_joint_list,
    separator,
    batched_list_to_split = None,
):
  """Yields student sublists from batched separator-joint multistudent lists.

  A batched implemention of per_student_iterator.
  Assumes the structure of then nested lists list_to_split and
  separator_joint_list is the same. batched_separator_joint_list must exactly 2
  levels of nesting.
  It assumes separator_joint_list contains lists of sequences of data,
  each corresponding to multiple students separated by entries equal to
  `separator`.


  For example if the first entries in batched_separator_joint_list
  and batched_list_to_split were

  batched_separator_joint_list[0]  =
          [ 1,    2, <separator>,    3, <separator>,   4,   5]
  batched_separator_joint_list[0]  =
          ['a', 'b',         'c',  'd',         'e', 'f', 'g']
  Then this generator would yield the following lists from the first entries
  ['a', 'b']
  ['d']
  ['f', 'g']

  before continuing to yield the remainder of the entries in a similar fashion.
  Args:
    batched_separator_joint_list: The nested list with separator tokens. This is
      used to infer the structure of list_to_split with regard to student
      sublists.
    separator: The separator token value.
    batched_list_to_split: The nested list to yield sublits from.  If None then
      separator_joint_list is used.

  Yields:
    sublists from elements of  list_to_split refering to single students.
    Separator tokens are removed from sublists before yielding.
  """

  if batched_list_to_split is None:
    batched_list_to_split = batched_separator_joint_list

  for separator_joint_list, list_to_split in zip(
      batched_separator_joint_list, batched_list_to_split
  ):
    yield from _per_student_iterator(
        separator_joint_list, list_to_split, separator
    )


def _unshard(tensor):
  """Inverse operation to common_utils.shard."""
  return tensor.reshape(-1, *tensor.shape[2:])


def _shard_and_pad(tensor):
  """Shard a tensor along the leading dimesion across available devices.

  If the size of the leading dimension tensor is not divisible by the number of
  available devices then the tensor will be padded.

  Args:
    tensor: tensor to shard and pad.

  Returns:
    Sharded and padded tensor.
  """
  if tensor.shape[0] % jax.local_device_count():
    remenant = -tensor.shape[0] % jax.local_device_count()
    tensor = jnp.concatenate([tensor, jnp.zeros_like(tensor[:remenant])])
  return common_utils.shard(tensor)


def recommend_from_datasource(
    eval_config,
    model_class,
    params,
    eval_datasource,
    n_recommendations,
    vocab,
    per_device_batch_size,
    n_logits_to_discard = 2,
):
  """Get predicted recommendations for data in a datasource.

  Run model inference to generate predictions from a dataset of historical data.

  Args:
    eval_config: TransformerConfig to use for inference.
    model_class: The class of the model to use for inference.
    params: The parameters of the transformer to use for inference.
    eval_datasource: The datasource to supply the data to use for inference.
    n_recommendations: Number of recommendations to produce at each timestep.
    vocab: The vocabulary to use to decode TokenIndices.
    per_device_batch_size: The maximum number of samples that we can run
      inference for on one TPU/GPU/Accelerator.
    n_logits_to_discard: How many trailing logits to discard. Can be useful to
      force the model to not predict out_of_vocabulary or separator token as as
      a recommendation.

  Returns:
    A dictionary mapping StudentIDs to a list of list of titles. The i'th list
      contains n_recommendations recommendations for the given student before
      they read the i'th title.
  """

  # This is necessary so we produce the recommendations in the correct order.
  assert eval_datasource.is_ordered_within_student()

  # Use a larger batch size if multiple accelartors are available.
  batch_size = per_device_batch_size * jax.local_device_count()

  data_iterator = datasource_lib.exhaustive_batch_sampler(
      eval_datasource, batch_size
  )
  params = jax_utils.replicate(params)

  p_infer_step = jax.pmap(
      functools.partial(
          _infer_step,
          config=eval_config,
          model_class=model_class,
          top_k=n_recommendations,
          n_logits_to_discard=n_logits_to_discard,
      )
  )

  # The indices correspond to studentID, timestep, n_recommenations
  results: list[list[list[str]]] = []
  # Correspoding studentID
  student_ids: list[types.StudentID] = []

  for i, (batch) in enumerate(data_iterator):
    logging.info('Running inference on batch no. %d', i)

    # prepare the batch for sharding across multiple devices.
    batch = jax.tree.map(_shard_and_pad, batch)

    # Run inference on all available accelerators. Get the top n_recommendation
    # titles at each timestep.
    inds = p_infer_step(params=params, inputs=batch)

    # Remove the dimension added by sharding
    inds = _unshard(inds)
    batch_student_ids = _unshard(batch[FIELDS.STUDENT_IDS])

    # Convert to numpy.ndarray and remove redundant dims.
    inds = np.array(inds).squeeze()

    # Batches containing short sequences have padding. In addition
    # shard_and_pad may have padded the final batch.  We create a padding_mask
    # to use so we filter out recommendations made on padding.
    padding_mask = (_unshard(batch[FIELDS.TITLES]) == 0)[:, :, None]
    padding_mask = np.asarray(padding_mask)

    # Decode titles so we have string SHLF_NUMs instead of types.TokenIndex.
    recommended_titles = np.vectorize(vocab.decode)(inds)

    # Tile (repeat) the padding_mask n_recommendations time to match the shape
    # recommended_titles.
    tile_required = [1] * len(recommended_titles.shape)
    tile_required[-1] = recommended_titles.shape[-1]
    padding_mask = np.tile(padding_mask, tile_required)

    # Use the padding_mask to remove padding recommendations.
    recommended_titles = np.ma.masked_array(
        recommended_titles, mask=padding_mask
    )

    titles = _unshard(batch[FIELDS.TITLES]).tolist()
    recommended_titles = recommended_titles.tolist()

    for student_recommendations in _batched_per_student_iterator(
        batched_separator_joint_list=titles,
        batched_list_to_split=recommended_titles,
        separator=vocab[input_pipeline_utils.SEPARATOR],
    ):
      # For each student sublist of recommendations  filter out padding
      # recommendations (which will be a list of None's).
      results.append(
          [
              time_stamp_recs
              for time_stamp_recs in student_recommendations
              if not any(rec is None for rec in time_stamp_recs)
          ]
      )

    for student_id_list in _batched_per_student_iterator(
        batch_student_ids.tolist(),
        vocab[input_pipeline_utils.SEPARATOR],
    ):
      student_ids.append(types.StudentID(student_id_list[0]))

  logging.info('Collating...')
  # Some students who have long interaction histories will be yielded in
  # chunks by the datasource across multiple different datapoints.
  # The following stanza compiles these into a single list per student.
  recommendations = collections.defaultdict(list)

  for student_recs, student_id in zip(results, student_ids):
    # filters out non-exisent students filtered with padding only interactions.
    if student_recs:
      recommendations[student_id].extend(student_recs)

  logging.info('Recommendations computed successfully.')
  return dict(recommendations)
