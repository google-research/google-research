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

"""Library for ETC sequence tagger."""

from typing import Any, Callable, Optional, Union

from absl import logging
import ml_collections
import tensorflow as tf

from etcmodel import feature_utils
from etcmodel.models import input_utils
from etcmodel.models import modeling

_ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]
_Initializer = Union[str, Any]


def make_global_local_transformer_side_inputs(
    long_paragraph_breakpoints,
    long_paragraph_ids,
    global_paragraph_breakpoints,
    local_radius,
    relative_pos_max_distance,
    use_hard_g2l_mask = False,
    use_hard_l2g_mask = False,
    name = None):
  """Makes attention masks and relative ids for l2l, l2g, g2g, g2l for QA tasks.

  Args:
    long_paragraph_breakpoints: <int32>[batch_size, global_seq_len] Tensor of
      `0`s and `1`s indicating paragraph boundaries in the long input.
    long_paragraph_ids: <int32>[batch_size, long_seq_len] Tensor of ids
      indicating the paragraph each token belongs to.
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
    use_hard_l2g_mask: If True, long tokens only attend to tokens of the
      corresponding global tokens. If False, long tokens attend to all the
      global tokens within the corresponding global example.
    name: A name for the operation (optional).

  Returns:
    A `GlobalLocalTransformerSideInputs` with all relevant tensors set.
  """
  with tf.name_scope(name or "make_global_local_transformer_side_inputs"):

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
        breakpoint_segments):
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

    global_range = tf.range(global_seq_len, dtype=long_paragraph_ids.dtype)

    if use_hard_g2l_mask:
      # Have each global token attend to just one paragraph instead of having
      # it attend to all the paragraphs within a global example.
      paragraph_hard_g2l_att_mask = tf.equal(
          global_range[tf.newaxis, :, tf.newaxis],
          long_paragraph_ids[:, tf.newaxis, :])

      # CLS token in the global will ignore the hard g2l mask. It should attend
      # to everything in the long.
      ignore_hard_g2l_att_mask = tf.tile(
          tf.sparse.to_dense(
              tf.SparseTensor([[0, 0, 0]], [True], [1, global_seq_len, 1])),
          [batch_size, 1, long_seq_len])

      hard_g2l_att_mask = tf.math.logical_or(paragraph_hard_g2l_att_mask,
                                             ignore_hard_g2l_att_mask)

      hard_g2l_att_mask = tf.cast(hard_g2l_att_mask, dtype=tf.int32)
      g2l_att_mask *= hard_g2l_att_mask

    if use_hard_l2g_mask:
      # Have each long token attend to just the corresponding global token
      # instead of having it attend to all the global tokens within a
      # global example.
      paragraph_hard_l2g_att_mask = tf.equal(
          long_paragraph_ids[:, :, tf.newaxis], global_range[tf.newaxis,
                                                             tf.newaxis, :])

      hard_l2g_att_mask = tf.cast(paragraph_hard_l2g_att_mask, dtype=tf.int32)
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

      # Add relative att ids for global paragraph level tokens.
      paragraph_l2g_relative_att_ids = tf.equal(
          global_range[tf.newaxis, tf.newaxis, :],
          long_paragraph_ids[:, :, tf.newaxis])

      l2g_relative_att_ids = tf.cast(
          paragraph_l2g_relative_att_ids, dtype=tf.int32)

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


class EtcSequenceTagger(tf.keras.Model):
  """ETC sequence tagger model."""

  def __init__(self,
               etc_config,
               use_one_hot_embeddings = False,
               use_one_hot_relative_embeddings = False,
               initializer = None):
    super(EtcSequenceTagger, self).__init__()
    self.etc_config = etc_config
    self.etc_model = modeling.EtcModel(
        etc_config,
        use_one_hot_embeddings=use_one_hot_embeddings,
        use_one_hot_relative_embeddings=use_one_hot_relative_embeddings)
    self.global_head = tf.keras.layers.Dense(
        1,  # This layer predicts token level binary label.
        kernel_initializer=initializer,
        name="predictions/transform/global_logits")
    self.long_head = tf.keras.layers.Dense(
        1,  # This layer predicts token level binary label.
        kernel_initializer=initializer,
        name="predictions/transform/long_logits")

  def call(self, inputs, training = False, mask = None):
    del mask  # Unused.
    etc_model_inputs = dict(
        token_ids=inputs["long_token_ids"],
        global_token_ids=inputs["global_token_ids"],
        segment_ids=inputs["long_token_type_ids"],
        global_segment_ids=inputs["global_token_type_ids"])
    etc_model_inputs.update(
        make_global_local_transformer_side_inputs(
            long_paragraph_breakpoints=inputs["long_breakpoints"],
            long_paragraph_ids=inputs["long_paragraph_ids"],
            global_paragraph_breakpoints=inputs["global_breakpoints"],
            local_radius=self.etc_config.local_radius,
            relative_pos_max_distance=self.etc_config.relative_pos_max_distance,
            use_hard_g2l_mask=self.etc_config.use_hard_g2l_mask,
            use_hard_l2g_mask=self.etc_config.use_hard_l2g_mask).to_dict(
                exclude_none_values=True))

    logging.info("######################ETC inputs################")
    for k, v in etc_model_inputs.items():
      logging.info(k)
      logging.info(v)

    long_output, global_output = self.etc_model(
        **etc_model_inputs, training=training)

    global_logits = self.global_head(global_output)
    long_logits = self.long_head(long_output)

    global_predictions = tf.nn.sigmoid(global_logits)
    long_predictions = tf.nn.sigmoid(long_logits)

    return {"global": global_predictions, "long": long_predictions}


def build_model(config):
  """Returns BERT sequence tagging model along with core BERT model."""
  etc_config = modeling.EtcConfig.from_json_file(config.etc.etc_config_file)
  initializer = tf.keras.initializers.TruncatedNormal(
      stddev=etc_config.initializer_range)

  model = EtcSequenceTagger(etc_config, config.etc.use_one_hot_embeddings,
                            config.etc.use_one_hot_relative_embeddings,
                            initializer)

  if config.etc.initial_checkpoint:
    # Not using tf.train.Checkpoint.read, since it loads Keras step counter.
    model.load_weights(config.etc.initial_checkpoint)
    logging.info("ETC tagger initialized from initial checkpoint: %s",
                 config.etc.initial_checkpoint)
  elif config.etc.pretrain_checkpoint:
    checkpoint = tf.train.Checkpoint(
        model=model.etc_model, encoder=model.etc_model)
    checkpoint.read(
        config.etc.pretrain_checkpoint).assert_existing_objects_matched()
    logging.info("ETC backbone initialized from pretrained checkpoint: %s",
                 config.etc.pretrain_checkpoint)

  # Builds the model by calling.
  model({
      "global_token_ids":
          tf.ones((1, config.etc.global_seq_length), tf.int32),
      "global_breakpoints":
          tf.ones((1, config.etc.global_seq_length), tf.int32),
      "global_token_type_ids":
          tf.ones((1, config.etc.global_seq_length), tf.int32),
      "long_token_ids":
          tf.ones((1, config.etc.long_seq_length), tf.int32),
      "long_breakpoints":
          tf.ones((1, config.etc.long_seq_length), tf.int32),
      "long_token_type_ids":
          tf.ones((1, config.etc.long_seq_length), tf.int32),
      "long_paragraph_ids":
          tf.ones((1, config.etc.long_seq_length), tf.int32),
  })
  return model
