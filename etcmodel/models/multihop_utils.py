# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utilities for ETC multi-hop question answer model inputs.

This is used for HotpotQA and WikiHop.
"""

from typing import List, Optional, Text

import attr
import tensorflow.compat.v1 as tf

from etcmodel import feature_utils
from etcmodel.models import input_utils


CLS_TOKEN_ID = 101

SENTENCE_GLOBAL_TOKEN_ID = 1
CPC_MASK_GLOBAL_TOKEN_ID = 2
QUESTION_GLOBAL_TOKEN_ID = 3
CANDIDATE_GLOBAL_TOKEN_ID = 4
PARAGRAPH_GLOBAL_TOKEN_ID = 5

SENTENCEPIECE_DEFAULT_GLOBAL_TOKEN_IDS = dict(
    CLS_TOKEN_ID=40,  # "<unused_39>"
    SENTENCE_GLOBAL_TOKEN_ID=1,  # "</s>", used in pretrianing.
    CPC_MASK_GLOBAL_TOKEN_ID=2,  # "<s>", used in pretrianing.
    QUESTION_GLOBAL_TOKEN_ID=41,  # "<unused_40>"
    CANDIDATE_GLOBAL_TOKEN_ID=42,  # "<unused_41>"
    PARAGRAPH_GLOBAL_TOKEN_ID=43,  # "<unused_42>"
)

GLOBAL_TOKEN_TYPE_ID = 0
SENTENCE_TOKEN_TYPE_ID = 1
TITLE_TOKEN_TYPE_ID = 2
QUESTION_TOKEN_TYPE_ID = 3
CANDIDATE_TOKEN_TYPE_ID = 4
QUESTION_GLOBAL_TOKEN_TYPE_ID = 5
CANDIDATE_GLOBAL_TOKEN_TYPE_ID = 6
PARAGRAPH_GLOBAL_TOKEN_TYPE_ID = 7


@attr.s
class InputConfig(object):
  """Config options for ETC QA model input."""
  # Sequence length of the global input.
  global_seq_length = attr.ib(type=int)
  # Sequence length of the long input.
  long_seq_length = attr.ib(type=int)
  # Whether in training mode.
  is_training = attr.ib(default=True, type=bool)
  # Whether the debug mode is on.
  debug = attr.ib(default=True, type=bool)
  # CLS token id.
  cls_token_id = attr.ib(default=101, type=int)

  # Global token ids. Global and long tokens share a same vocab.
  # Sentence global token id. Using 1 for transfer learning.
  sentence_global_token_id = attr.ib(default=SENTENCE_GLOBAL_TOKEN_ID, type=int)
  # CPC mask global token id. Using 2 for transfer learning.
  cpc_mask_global_token_id = attr.ib(default=CPC_MASK_GLOBAL_TOKEN_ID, type=int)
  # Question global token id.
  question_global_token_id = attr.ib(default=QUESTION_GLOBAL_TOKEN_ID, type=int)
  # Candidate global token id.
  candidate_global_token_id = attr.ib(
      default=CANDIDATE_GLOBAL_TOKEN_ID, type=int)
  # Paragraph global token id.
  paragraph_global_token_id = attr.ib(
      default=PARAGRAPH_GLOBAL_TOKEN_ID, type=int)

  # Token type ids. Global and long token types share a same vocab of size 16.
  # Global token type id. Using 0 for transfer learning.
  global_token_type_id = attr.ib(default=GLOBAL_TOKEN_TYPE_ID, type=int)
  # Sentence token type id.
  sentence_token_type_id = attr.ib(default=SENTENCE_TOKEN_TYPE_ID, type=int)
  # Title token type id.
  title_token_type_id = attr.ib(default=TITLE_TOKEN_TYPE_ID, type=int)
  # question token type id.
  question_token_type_id = attr.ib(default=QUESTION_TOKEN_TYPE_ID, type=int)
  # Candidate token type id.
  candidate_token_type_id = attr.ib(default=CANDIDATE_TOKEN_TYPE_ID, type=int)


@attr.s
class InputFeatures(object):
  """The feautres for ETC QA model."""
  # Context features
  # Long token ids with format question paragraph1 paragraph2 ...
  long_token_ids = attr.ib(factory=list, type=List[int])
  # The sentence ids for the long tokens. Each id `i` corresponds to a sentence
  # global token, `global_token_ids[i]`. Each question token has a
  # unique sentence id.
  long_sentence_ids = attr.ib(factory=list, type=List[int])
  # The paragraph ids for the long tokens. Each id `i` corresponds to a
  # paragraph global token, `global_token_ids[i]`. Question tokens don't
  # correspond to any paragraph global tokens.
  long_paragraph_ids = attr.ib(factory=list, type=List[int])
  # Ending breakpoints separating question and paragraphs long tokens into
  # different segments, which are not attended by local attentions.
  long_paragraph_breakpoints = attr.ib(factory=list, type=List[int])
  # The token type ids for long tokens. With default values in `InputConfig`.
  long_token_type_ids = attr.ib(factory=list, type=List[int])
  # The global token ids. With default values in `InputConfig`.
  global_token_ids = attr.ib(factory=list, type=List[int])
  # Ending breakpoints separating question and paragraphs global
  # tokens into different segments.
  global_paragraph_breakpoints = attr.ib(factory=list, type=List[int])
  # The token type ids for global tokens. With default values in `InputConfig`.
  global_token_type_ids = attr.ib(factory=list, type=List[int])
  # Flag to indicate whether this is a real / padding example. Padding examples
  # can be useful to pad up to a multiple of batch_size examples. This can
  # be helpful in cases where TPUs require fixed batch_size (esp. for eval /
  # predict). For training, it can help us so that we don't drop remainder
  # (num_examples % batch_size) examples.
  is_real_example = attr.ib(default=True)  # type: bool


def make_global_local_transformer_side_inputs(
    long_paragraph_breakpoints: tf.Tensor,
    long_paragraph_ids: tf.Tensor,
    long_sentence_ids: tf.Tensor,
    global_paragraph_breakpoints: tf.Tensor,
    local_radius: int,
    relative_pos_max_distance: int,
    use_hard_g2l_mask: bool = False,
    ignore_hard_g2l_mask: tf.Tensor = None,
    use_hard_l2g_mask: bool = False,
    ignore_hard_l2g_mask: tf.Tensor = None,
    flat_sequence: bool = False,
    l2g_linked_ids: Optional[tf.Tensor] = None,
    name: Optional[Text] = None
) -> input_utils.GlobalLocalTransformerSideInputs:
  """Makes attention masks and relative ids for l2l, l2g, g2g, g2l for QA tasks.

  When `use_hard_g2l_mask=True` and `use_hard_l2g_mask=False`, the resulting
  attention pattern is similar to Figure 3b of the paper for representing
  a set of (unordered) contexts ("paragraphs" here), except instead of
  defining a new relative position label between a global paragraph token and
  its global sentence tokens, we just place each global paragraph token as
  the first token before subsequent global sentence tokens belonging to it.

  Note: This function assumes that we don't pack multiple examples into a single
  example, which is only done for pre-training.

  See `GlobalLocalTransformerLayers.call()` in `layers/transformer.py` for a
  description of the 8 side inputs.

  Args:
    long_paragraph_breakpoints: <int32>[batch_size, global_seq_len] Tensor of
      `0`s and `1`s indicating paragraph boundaries in the long input.
    long_paragraph_ids: <int32>[batch_size, long_seq_len] Tensor of ids
      indicating the paragraph each token belongs to.
    long_sentence_ids: <int32>[batch_size, long_seq_len] Tensor of ids
      indicating which sentence each token belongs to.
    global_paragraph_breakpoints: <int32>[batch_size, global_seq_len] Tensor of
      of `0`s and `1`s indicating paragraph boundaries in the global input.
    local_radius: How many tokens to the left/right for input tokens to locally
      self-attend to. For example, a value of 1 would allow each token to only
      attend to 1 token to the left and 1 token to the right of it.
    relative_pos_max_distance: Maximum distance to use for relative position
      representations. All larger distances will be clipped to this value. Use 0
      to skip relative position representations entirely.
    use_hard_g2l_mask: If True, global tokens only attend to tokens of the
      corresponding sentences in the long input. If False, global tokens attend
      to all sentences within the corresponding global example.
    ignore_hard_g2l_mask: <int32>[batch_size, global_seq_len] Tensor of `0`s and
      `1`s indicating the indices in the global input which should ignore the
      `use_hard_g2l_mask`. `1` is for ignoring the hard mask and these tokens
      essentially attend to everything (except for padding tokens) in the long
      input. This can be useful to force some tokens (e.g, CLS) to attend to
      everything in the long input even though they don't necessarily map to
      anything in the long input via sentence / paragraph ids etc. This tensor
      will be applicable only when `use_hard_g2l` is enabled.
    use_hard_l2g_mask: If True, long tokens only attend to tokens of the
      corresponding global tokens. If False, long tokens attend to all the
      global tokens within the corresponding global example.
    ignore_hard_l2g_mask: <int32>[batch_size, long_seq_len] Tensor of `0`s and
      `1`s indicating the indices in the long input which should ignore the
      `use_hard_l2g_mask`. `1` is for ignoring the hard mask and these tokens
      essentially attend to everything (except for padding tokens) in the global
      input. This can be useful to force some tokens (e.g, query tokens) to
      attend to everything in the global input even though they don't
      necessarily map to anything in the global input via sentence / paragraph
      ids etc. This tensor will be applicable only when `use_hard_l2g` is
      enabled.
    flat_sequence: If True, the attention masks / relative attention ids would
      be computing assuming the default ETC setting where there is not any
      structure (except for having the notion of a "sentence").
    l2g_linked_ids: <int32>[batch_size, long_seq_len] Tensor specifying the long
      tokens which should be linked to the global tokens. If the input is [[-1,
      -1, 0, 1, 1, -1]], then 2nd long token would be linked to 0-th global
      token and 3rd, 4-th long tokens woulbe linked to the 1st global token.
    name: A name for the operation (optional).

  Returns:
    A `GlobalLocalTransformerSideInputs` with all relevant tensors set.
  """
  with tf.name_scope(name or 'make_global_local_transformer_side_inputs'):

    long_input_mask = tf.minimum(
        tf.cumsum(long_paragraph_breakpoints, axis=-1, reverse=True), 1)
    global_input_mask = tf.minimum(
        tf.cumsum(global_paragraph_breakpoints, axis=-1, reverse=True), 1)

    if flat_sequence:
      # Here we don't use any structure in the input i.e it falls back to
      # the default ETC setting where:
      # a) everything in the long can attend to everything in the global and
      #    vice-versa.
      # b) everything in global attends to everything in global.
      # c) everything in long can attend to everything in long that is within
      #    the local radius
      #
      # Note that there is a small caveat here: The paragraph / cls level tokens
      # in the global input would be orphaned (i.e they wouldn't be linked to
      # anything in the long), but that should be probably
      # okay as they still attend to everything in the global.
      #
      # We don't have any packing here. So we need to construct
      # long/global breakpoints to indicate there's only one example.
      # The structure of these breakpoints should be as follows:
      # [0, 0, .....,1, 0, 0, 0] i.e there should be a single `1` just before
      # the padding begins, rest of the tokens should be `0`.
      return (input_utils
              .make_global_local_transformer_side_inputs_from_example_ids(
                  long_example_ids=long_input_mask,
                  global_example_ids=global_input_mask,
                  sentence_ids=long_sentence_ids,
                  local_radius=local_radius,
                  relative_pos_max_distance=relative_pos_max_distance,
                  use_hard_g2l_mask=use_hard_g2l_mask,
                  use_hard_l2g_mask=use_hard_l2g_mask))

    # Make paragraphs not attend to other paragraphs in the long input.
    long_paragraph_breakpoints = tf.convert_to_tensor(
        long_paragraph_breakpoints)
    long_paragraph_breakpoint_segments = tf.cumsum(
        long_paragraph_breakpoints, axis=-1, reverse=True)

    l2l_att_mask = feature_utils.make_local_segmented_att_mask(
        long_paragraph_breakpoint_segments, local_radius)

    global_paragraph_breakpoints = tf.convert_to_tensor(
        global_paragraph_breakpoints)
    global_paragraph_breakpoint_segments = tf.cumsum(
        global_paragraph_breakpoints, axis=-1, reverse=True)

    # For g2l, g2g and l2g, we can have everything attend everything else.
    # So we can have attention tokens as all `1`s and account for padding via
    # a mask.
    def _make_input_mask_from_breakpoints(
        breakpoint_segments: tf.Tensor) -> tf.Tensor:
      return tf.minimum(
          tf.cast(1, dtype=breakpoint_segments.dtype), breakpoint_segments)

    long_attention_tokens = _make_input_mask_from_breakpoints(
        long_paragraph_breakpoint_segments)

    # Ignore the padding tokens.
    global_attention_tokens = _make_input_mask_from_breakpoints(
        global_paragraph_breakpoint_segments)

    g2g_att_mask = feature_utils.make_segmented_att_mask(
        global_attention_tokens)
    l2g_att_mask = tf.cast(
        tf.equal(long_attention_tokens[:, :, tf.newaxis],
                 global_attention_tokens[:, tf.newaxis, :]), tf.int32)
    g2l_att_mask = tf.transpose(l2g_att_mask, perm=[0, 2, 1])

    long_seq_len = long_paragraph_breakpoints.shape.as_list()[1]
    assert long_seq_len is not None

    global_seq_len = global_paragraph_breakpoints.shape.as_list()[1]
    assert global_seq_len is not None

    batch_size = tf.shape(long_paragraph_breakpoints)[0]
    assert batch_size is not None

    global_range = tf.range(global_seq_len, dtype=long_sentence_ids.dtype)
    long_ones = tf.ones_like(long_sentence_ids)
    global_ones = tf.ones_like(global_paragraph_breakpoints)

    if use_hard_g2l_mask:
      if ignore_hard_g2l_mask is None:
        ignore_hard_g2l_mask = tf.zeros_like(global_paragraph_breakpoints)
      else:
        ignore_hard_g2l_mask = tf.convert_to_tensor(ignore_hard_g2l_mask)

      # Have each global token attend to just one sentence instead of having
      # it attend to all the sentences within a global example.
      sentence_hard_g2l_att_mask = tf.equal(
          global_range[tf.newaxis, :, tf.newaxis],
          long_sentence_ids[:, tf.newaxis, :])

      # Also have paragraph global tokens attend to the corresponding long
      # paragraphs.
      paragraph_hard_g2l_att_mask = tf.equal(
          global_range[tf.newaxis, :, tf.newaxis],
          long_paragraph_ids[:, tf.newaxis, :])

      ignore_hard_g2l_att_mask = tf.equal(
          ignore_hard_g2l_mask[:, :, tf.newaxis], long_ones[:, tf.newaxis, :])

      # It's possible that certain global tokens, although linked to a long
      # sentence, might still be present in `ignore_hard_g2l_mask`. Such tokens
      # should also attend to everything in the long.
      hard_g2l_att_mask = tf.math.logical_or(
          tf.math.logical_or(sentence_hard_g2l_att_mask,
                             paragraph_hard_g2l_att_mask),
          ignore_hard_g2l_att_mask)

      hard_g2l_att_mask = tf.cast(hard_g2l_att_mask, dtype=tf.int32)
      g2l_att_mask *= hard_g2l_att_mask

    if use_hard_l2g_mask:
      if ignore_hard_l2g_mask is None:
        ignore_hard_l2g_mask = tf.zeros_like(long_sentence_ids)
      else:
        ignore_hard_l2g_mask = tf.convert_to_tensor(ignore_hard_l2g_mask)

      # Have each long token attend to just the corresponding global token
      # instead of having it attend to all the global tokens within a
      # global example.
      sentence_hard_l2g_att_mask = tf.equal(
          long_sentence_ids[:, :, tf.newaxis], global_range[tf.newaxis,
                                                            tf.newaxis, :])

      # Also have paragraph global tokens attend to the corresponding long
      # paragraphs.
      paragraph_hard_l2g_att_mask = tf.equal(
          long_paragraph_ids[:, :, tf.newaxis], global_range[tf.newaxis,
                                                             tf.newaxis, :])

      ignore_hard_l2g_att_mask = tf.equal(
          ignore_hard_l2g_mask[:, :, tf.newaxis], global_ones[:, tf.newaxis, :])

      # It's possible that certain long tokens, although linked to global tokens
      # might still be present in `ignore_hard_l2g_mask`. Such tokens
      # should also attend to everything in the global.
      hard_l2g_att_mask = tf.math.logical_or(
          tf.math.logical_or(sentence_hard_l2g_att_mask,
                             paragraph_hard_l2g_att_mask),
          ignore_hard_l2g_att_mask)

      hard_l2g_att_mask = tf.cast(hard_l2g_att_mask, dtype=tf.int32)
      l2g_att_mask *= hard_l2g_att_mask

    l2l_relative_att_ids = None
    g2g_relative_att_ids = None
    l2g_relative_att_ids = None
    g2l_relative_att_ids = None

    if relative_pos_max_distance > 0:

      relative_pos_generator = feature_utils.RelativePositionGenerator(
          relative_pos_max_distance)

      l2l_relative_att_ids = relative_pos_generator.make_local_relative_att_ids(
          seq_len=long_seq_len,
          local_radius=local_radius,
          batch_size=batch_size)

      sentence_l2g_relative_att_ids = tf.equal(
          long_sentence_ids[:, :, tf.newaxis], global_range[tf.newaxis,
                                                            tf.newaxis, :])

      # Add relative att ids for global paragraph level tokens.
      paragraph_l2g_relative_att_ids = tf.equal(
          global_range[tf.newaxis, tf.newaxis, :],
          long_paragraph_ids[:, :, tf.newaxis])

      if l2g_linked_ids is None:
        l2g_linked_relative_att_ids = tf.zeros_like(
            paragraph_l2g_relative_att_ids)
      else:
        l2g_linked_ids = tf.convert_to_tensor(l2g_linked_ids)
        l2g_linked_relative_att_ids = tf.equal(
            global_range[tf.newaxis, tf.newaxis, :], l2g_linked_ids[:, :,
                                                                    tf.newaxis])

      l2g_relative_att_ids = tf.cast(
          tf.math.logical_or(
              l2g_linked_relative_att_ids,
              tf.math.logical_or(sentence_l2g_relative_att_ids,
                                 paragraph_l2g_relative_att_ids)),
          dtype=tf.int32)

      g2l_relative_att_ids = tf.transpose(l2g_relative_att_ids, perm=[0, 2, 1])

      # For fused attention, l2l and l2g share the same relative vocabulary, as
      # do g2g and g2l, so we add an offset for l2g and g2l so their original
      # 0/1 ids don't collide with l2l and g2g relative position ids.
      l2g_relative_att_ids += relative_pos_generator.relative_vocab_size
      g2l_relative_att_ids += relative_pos_generator.relative_vocab_size

      g2g_relative_att_ids = relative_pos_generator.make_relative_att_ids(
          seq_len=global_seq_len, batch_size=batch_size)

      # We used up 2 ids to account for the collision in fused attention as
      # mentioned above. Hence the +2.
      g2g_max_rel_id = relative_pos_generator.relative_vocab_size + 2
      g2g_relative_att_ids = (
          feature_utils.overwrite_relative_att_ids_outside_segments(
              rel_att_ids=g2g_relative_att_ids,
              segment_ids=global_paragraph_breakpoint_segments,
              overwrite_value=g2g_max_rel_id))

    return input_utils.GlobalLocalTransformerSideInputs(
        l2l_att_mask=l2l_att_mask,
        g2g_att_mask=g2g_att_mask,
        l2g_att_mask=l2g_att_mask,
        g2l_att_mask=g2l_att_mask,
        l2l_relative_att_ids=l2l_relative_att_ids,
        g2g_relative_att_ids=g2g_relative_att_ids,
        l2g_relative_att_ids=l2g_relative_att_ids,
        g2l_relative_att_ids=g2l_relative_att_ids)
