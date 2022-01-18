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

"""Utilities for featurizing inputs.

This includes utilities for generating attention masks and relative attention
ids which correspond to `att_mask` and `relative_att_ids` arguments,
respectively, for the attention and transformer layers in the `layers` folder.
"""

from typing import Optional, Text, Union

import tensorflow as tf

from etcmodel import tensor_utils


class RelativePositionGenerator(object):
  """Generates `relative_att_ids` for purely distance-based relative positions.

  This implements the clipped relative position representations originally
  described in https://arxiv.org/abs/1803.02155 .

  Attributes:
    max_distance: Integer passed from `__init__`.
    ignore_direction: Bool passed from `__init__`.
    relative_vocab_size: Integer representing the maximum number of unique ids
      output from this generator.
    left_pad_value: Integer id for all positions at or beyond max_distance to
      the left.
    right_pad_value: Integer id for all positions at or beyond max_distance to
      the right.
  """

  def __init__(self, max_distance: int, ignore_direction: bool = False):
    """Init.

    Args:
      max_distance: The maximum distance to represent. Must not be negative. All
        larger distances will be clipped to this value.
      ignore_direction: If True, both left and right position representations
        will have the same ids based on absolute distance (resulting in
        symmetric ids around the center token).
    """
    if max_distance < 0:
      raise ValueError('`max_distance` must not be negative.')
    self.max_distance = max_distance
    self.ignore_direction = ignore_direction

    self.right_pad_value = max_distance
    self.left_pad_value = max_distance if ignore_direction else 2 * max_distance

    # 0 is the first id, so vocab size is 1 + the largest id (left pad value).
    self.relative_vocab_size = self.left_pad_value + 1

  def make_relative_att_ids(self,
                            seq_len: Union[int, tf.Tensor],
                            batch_size: Optional[Union[int, tf.Tensor]] = 1,
                            name: Optional[Text] = None) -> tf.Tensor:
    """Makes relative position ids for full self-attention.

    For example, if `max_distance` is 3, `ignore_direction` is False, `seq_len`
    is 6, and `batch_size` is 1, the result is the following:
      [[
          [0, 1, 2, 3, 3, 3],
          [4, 0, 1, 2, 3, 3],
          [5, 4, 0, 1, 2, 3],
          [6, 5, 4, 0, 1, 2],
          [6, 6, 5, 4, 0, 1],
          [6, 6, 6, 5, 4, 0],
      ]]

    Args:
      seq_len: The sequence length to create ids for. Must be positive. If a
        Tensor, must be a scalar int.
      batch_size: The batch size of the result (default 1). Must be positive. If
        a Tensor, must be a scalar int. All examples in the batch will have the
        same id pattern.
      name: A name for the operation (optional).

    Returns:
      <int32>[batch_size, seq_len, seq_len] Tensor of relative position ids.
    """
    with tf.name_scope(name or 'make_relative_att_ids'):
      if isinstance(seq_len, int) and seq_len < 1:
        raise ValueError('`seq_len` must be positive.')
      if isinstance(batch_size, int) and batch_size < 1:
        raise ValueError('`batch_size` must be positive.')

      # We need the id_pattern to cover all tokens to the left of the last token
      # and all tokens to the right of the first token at the same time.
      window_size = 2 * seq_len - 1

      # [window_size]
      id_pattern = self._make_relative_id_pattern(window_size)

      # [seq_len, window_size]
      id_tensor = tf.tile(id_pattern[tf.newaxis, :], [seq_len, 1])

      # [seq_len, window_size + seq_len - 1]
      id_tensor = tensor_utils.skew_elements_right(id_tensor, -1)

      # [seq_len, seq_len]
      id_tensor = tf.slice(id_tensor, [0, seq_len - 1], [seq_len, seq_len])

      return tf.tile(id_tensor[tf.newaxis, :, :], [batch_size, 1, 1])

  def make_local_relative_att_ids(self,
                                  seq_len: Union[int, tf.Tensor],
                                  local_radius: int,
                                  batch_size: Optional[Union[int,
                                                             tf.Tensor]] = 1,
                                  name: Optional[Text] = None) -> tf.Tensor:
    """Makes relative position ids for local self-attention.

    The result can be used as `l2l_relative_att_ids` in
    `layers.GlobalLocalTransformerLayers`.

    For example, if `max_distance` is 3, `ignore_direction` is False, `seq_len`
    is 4, `local_radius` is 5, and `batch_size` is 1, the result is the
    following:
      [[
          [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],
          [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],
          [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],
          [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],
      ]]

    Args:
      seq_len: The sequence length to create ids for. Must be positive. If a
        Tensor, must be a scalar int.
      local_radius: The local radius as expected by
        `layers.GlobalLocalTransformerLayers`. Must be positive.
      batch_size: The batch size of the result (default 1). Must be positive. If
        a Tensor, must be a scalar int. All examples in the batch will have the
        same id pattern.
      name: A name for the operation (optional).

    Returns:
      <int32>[batch_size, seq_len, 2*local_radius + 1] Tensor of relative
      position ids.
    """
    with tf.name_scope(name or 'make_local_relative_att_ids'):
      if isinstance(seq_len, int) and seq_len < 1:
        raise ValueError('`seq_len` must be positive.')
      if local_radius < 1:
        raise ValueError('`local_radius` must be positive.')
      if isinstance(batch_size, int) and batch_size < 1:
        raise ValueError('`batch_size` must be positive.')

      window_size = 2 * local_radius + 1

      # [window_size]
      id_pattern = self._make_relative_id_pattern(window_size)

      return tf.tile(id_pattern[tf.newaxis, tf.newaxis, :],
                     [batch_size, seq_len, 1])

  def _make_relative_id_pattern(
      self, window_size: Union[int, tf.Tensor]) -> tf.Tensor:
    """Helper for making the relative id pattern for a particular window size.

    For example, if `max_distance` is 3, `ignore_direction` is False, and
    `window_size` is 11, the result is the following:
    [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3].

    Args:
      window_size: Window size to return relative ids for. Must be positive and
        odd since ids will be relative to the center of the window. If a Tensor,
        must be a scalar int.

    Returns:
      <int32>[window_size] Tensor of relative position ids.
    """
    if isinstance(window_size, int):
      if window_size < 1:
        raise ValueError('`window_size` must be positive.')
      if window_size % 2 != 1:
        raise ValueError('`window_size` must be odd.')

    x = tf.range(self.max_distance + 1, dtype=tf.int32)
    x = tf.pad(x, [[self.max_distance, 0]], mode='REFLECT')
    if not self.ignore_direction:
      direction_adder = tf.concat([
          tf.fill([self.max_distance], self.max_distance),
          tf.zeros([self.max_distance + 1], dtype=tf.int32)
      ], 0)
      x += direction_adder

    len_x = x.shape.as_list()[0]
    if len_x > window_size:
      trim_amount = (len_x - window_size) // 2
      return x[trim_amount:-trim_amount]

    pad_amount = (window_size - len_x) // 2
    result = tf.pad(x, [[pad_amount, 0]], constant_values=self.left_pad_value)
    result = tf.pad(
        result, [[0, pad_amount]], constant_values=self.right_pad_value)
    return result


def overwrite_relative_att_ids_outside_segments(
    rel_att_ids: tf.Tensor,
    segment_ids: tf.Tensor,
    overwrite_value: int,
    name: Optional[Text] = None) -> tf.Tensor:
  """Modifies the given relative position ids from attending across segments.

  Example:
  Let's say we have `seq_len` 8, `max_distance` of 3 and `rel_att_ids` as
  below:
    [[[
        [0, 1, 2, 3, 3, 3, 3, 3],
        [4, 0, 1, 2, 3, 3, 3, 3],
        [5, 4, 0, 1, 2, 3, 3, 3],
        [6, 5, 4, 0, 1, 2, 3, 3],
        [6, 6, 5, 4, 0, 1, 2, 3],
        [6, 6, 6, 5, 4, 0, 1, 2],
        [6, 6, 6, 6, 5, 4, 0, 1],
        [6, 6, 6, 6, 6, 5, 4, 0],
    ]]]

  Let's say the segment_ids = [10, 10, 10, 20, 20, 30, 40, 40] i.e
  the first 3 tokens belong to a one segment_id = 10, next two tokens
  belong to segment_id = 20, next token belongs to segment_id = 30 and the
  last two tokens belong to segment_id = 40. Because we don't want to attend
  across segment_ids, rel_att_ids would be modified and returned as below (
  where the overwrite_value = 8):
    [[[0, 1, 2, 8, 8, 8, 8, 8],
      [4, 0, 1, 8, 8, 8, 8, 8],
      [5, 4, 0, 8, 8, 8, 8, 8],
      [8, 8, 8, 0, 1, 8, 8, 8],
      [8, 8, 8, 4, 0, 8, 8, 8],
      [8, 8, 8, 8, 8, 0, 8, 8],
      [8, 8, 8, 8, 8, 8, 0, 1],
      [8, 8, 8, 8, 8, 8, 4, 0]]]

  Args:
    rel_att_ids: <int32>[batch_size, seq_len, seq_len] The relative attention
      ids with (potentially) tokens cross attending segments.
    segment_ids: <int32>[batch_size, seq_len] Tensor of segment ids.
    overwrite_value: The relative attention id to be used for tokens in the
      outside segments for a given token.
    name: A name for the operation (optional).

  Returns:
    <int32>[batch_size, seq_len, seq_len] Tensor of the new
    relative position ids.
  """
  with tf.name_scope(name or 'overwrite_relative_att_ids_outside_segments'):
    rel_att_ids = tf.convert_to_tensor(rel_att_ids)
    segment_ids = tf.convert_to_tensor(segment_ids)
    segment_mask = make_segmented_att_mask(segment_ids)
    new_rel_att_ids = tf.where(
        tf.equal(segment_mask, 1), rel_att_ids,
        tf.fill(tf.shape(rel_att_ids), overwrite_value))
    return new_rel_att_ids


def make_att_mask_from_input_mask(input_mask: tf.Tensor,
                                  name: Optional[Text] = None) -> tf.Tensor:
  """Makes a self-attention mask for the given input mask.

  The returned mask will allow all "unmasked" tokens to attend only to other
  unmasked tokens. Note we're not using "mask" in the "masked language model"
  sense here. The "masked" tokens are simply tokens corresponding to 0 padding.

  Args:
    input_mask: <int32>[batch_size, seq_len] input mask, with 1 for valid tokens
      and 0 for tokens to mask.
    name: A name for the operation (optional).

  Returns:
    <int32>[batch_size, seq_len, seq_len] attention mask.
  """
  # We just use `make_segmented_att_mask` to constrain attention within unmasked
  # tokens. This will allow all masked tokens to attend (only) to other masked
  # tokens also, but this is harmless since masked token results won't be used.
  with tf.name_scope(name or 'make_att_mask_from_input_mask'):
    return make_segmented_att_mask(input_mask)


def make_segmented_att_mask(segment_ids: tf.Tensor,
                            name: Optional[Text] = None) -> tf.Tensor:
  """Makes self-attention mask preventing attention across different segments.

  Restricts full self-attention to attend within segments. The tokens in a
  segment do not all need to be contiguous. All tokens can (only) attend to all
  other tokens from the same segment id.

  Args:
    segment_ids: <int32>[batch_size, seq_len] Tensor of segment ids, all of
      which must be non-negative.
    name: A name for the operation (optional).

  Returns:
    <int32>[batch_size, seq_len, seq_len] attention mask.
  """
  with tf.name_scope(name or 'make_segmented_att_mask'):
    segment_ids = tf.convert_to_tensor(segment_ids)

    if segment_ids.shape.rank != 2:
      raise ValueError('`segment_ids` must be a 2-D tensor.')

    return tf.cast(
        tf.equal(segment_ids[:, :, tf.newaxis], segment_ids[:, tf.newaxis, :]),
        tf.int32)


def make_att_mask_from_breakpoints(att_breakpoints: tf.Tensor,
                                   use_starting_breakpoints: bool = False,
                                   name: Optional[Text] = None) -> tf.Tensor:
  """Makes self-attention mask from attention breakpoints.

  Each attention breakpoint marks the end of a segment by default (or the
  start if `use_starting_breakpoints` is True), and the resulting
  mask prevents attention across different segments.

  Args:
    att_breakpoints: <int32>[batch_size, seq_len] Tensor containing only 0 and 1
      values, where each "1" marks the end of a segment (or the start, depending
      on `use_starting_breakpoints`).
    use_starting_breakpoints: If True, breakpoints represent starts of segments
      rather than ends of segments. Default False.
    name: A name for the operation (optional).

  Returns:
    <int32>[batch_size, seq_len, seq_len] attention mask.
  """
  with tf.name_scope(name or 'make_att_mask_from_breakpoints'):
    att_breakpoints = tf.convert_to_tensor(att_breakpoints)

    if att_breakpoints.shape.rank != 2:
      raise ValueError('`att_breakpoints` must be a 2-D tensor.')

    if not use_starting_breakpoints:
      att_breakpoints = tensor_utils.shift_elements_right(
          att_breakpoints, axis=-1, amount=1)

    segment_ids = tf.cumsum(att_breakpoints, axis=1)
    return make_segmented_att_mask(segment_ids)


def make_local_segmented_att_mask(segment_ids: tf.Tensor,
                                  local_radius: int,
                                  name: Optional[Text] = None) -> tf.Tensor:
  """Makes local attention mask preventing attention across different segments.

  Restricts local self-attention to attend within segments, such that tokens can
  only attend to local tokens from the same segment id. The tokens in a segment
  do not need to be contiguous, but attention is still constrained by
  `local_radius`. The output can be used as `l2l_att_mask` in
  `layers.GlobalLocalTransformerLayers` for example.

  Args:
    segment_ids: <int32>[batch_size, seq_len] Tensor of segment ids, all of
      which must be non-negative.
    local_radius: The local radius as expected by
      `layers.GlobalLocalTransformerLayers`. Must be positive.
    name: A name for the operation (optional).

  Returns:
    <int32>[batch_size, seq_len, 2*local_radius + 1] attention mask.
  """
  with tf.name_scope(name or 'make_local_segmented_att_mask'):
    segment_ids = tf.convert_to_tensor(segment_ids)

    if segment_ids.shape.rank != 2:
      raise ValueError('`segment_ids` must be a 2-D tensor.')

    batch_size, seq_len = tensor_utils.get_shape_list(segment_ids)

    # Add 1 so that segment id `0` doesn't coincide with `0` padding values
    # introduced later by `tensor_utils.concat_3_blocks()` for example.
    segment_ids += 1

    # [batch_size, num_blocks, local_radius]
    blocked_segment_ids = tensor_utils.split_into_blocks(
        segment_ids, block_len=local_radius, axis=1)

    # [batch_size, num_blocks, 3*local_radius]
    concat_blocked_segment_ids = tensor_utils.concat_3_blocks(
        blocked_segment_ids)

    # [batch_size, num_blocks, local_radius, 3*local_radius]
    tiled_segment_ids = tf.tile(concat_blocked_segment_ids[:, :, tf.newaxis, :],
                                [1, 1, local_radius, 1])

    # [batch_size, num_blocks, local_radius, 2*local_radius + 1]
    blocked_unskewed_segment_ids = tensor_utils.unskew_elements_right(
        tiled_segment_ids, axis=-1)

    # [batch_size, num_blocks * local_radius, 2*local_radius + 1]
    flat_unskewed_segment_ids = tensor_utils.flatten_dims(
        blocked_unskewed_segment_ids, first_dim=1, last_dim=2)

    # [batch_size, seq_len, 2*local_radius + 1]
    unskewed_segment_ids = tf.slice(
        flat_unskewed_segment_ids, begin=[0, 0, 0], size=[-1, seq_len, -1])

    # [batch_size, seq_len, 1]
    center_token_segment_id = unskewed_segment_ids[:, :,
                                                   local_radius:(local_radius +
                                                                 1)]

    # [batch_size, seq_len, 2*local_radius + 1]
    result = tf.cast(
        tf.equal(unskewed_segment_ids, center_token_segment_id), tf.int32)

    # Use `reshape` to set the static shape when known.
    return tf.reshape(result, [batch_size, seq_len, 2 * local_radius + 1])


def make_local_att_mask_from_breakpoints(
    att_breakpoints: tf.Tensor,
    local_radius: int,
    use_starting_breakpoints: bool = False,
    name: Optional[Text] = None) -> tf.Tensor:
  """Makes local self-attention mask from attention breakpoints.

  Each attention breakpoint marks the end of a segment by default (or the
  start if `use_starting_breakpoints` is True), and the resulting
  mask prevents attention across different segments. The result can be used as
  `l2l_att_mask` in `layers.GlobalLocalTransformerLayers` for example.

  Args:
    att_breakpoints: <int32>[batch_size, seq_len] Tensor containing only 0 and 1
      values, where each "1" marks the end of a segment (or the start, depending
      on `use_starting_breakpoints`).
    local_radius: The local radius as expected by
      `layers.GlobalLocalTransformerLayers`. Must be positive.
    use_starting_breakpoints: If True, breakpoints represent starts of segments
      rather than ends of segments. Default False.
    name: A name for the operation (optional).

  Returns:
    <int32>[batch_size, seq_len, 2*local_radius + 1] attention mask.
  """
  with tf.name_scope(name or 'make_local_att_mask_from_breakpoints'):
    att_breakpoints = tf.convert_to_tensor(att_breakpoints)

    if att_breakpoints.shape.rank != 2:
      raise ValueError('`att_breakpoints` must be a 2-D tensor.')

    if not use_starting_breakpoints:
      att_breakpoints = tensor_utils.shift_elements_right(
          att_breakpoints, axis=-1, amount=1)

    # [batch_size, seq_len]
    segment_ids = tf.cumsum(att_breakpoints, axis=1)

    return make_local_segmented_att_mask(segment_ids, local_radius)
