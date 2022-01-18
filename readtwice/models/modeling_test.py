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

"""Tests for modeling library."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from readtwice.models import config as model_config
from readtwice.models import modeling


class ReadItTwiceBertModelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="default_no_att_mask",
          use_att_mask=False,
          second_read_type="from_scratch",
          summary_postprocessing_type="none"),
      dict(
          testcase_name="default",
          second_read_type="from_scratch",
          summary_postprocessing_type="none"),
      dict(
          testcase_name="dont_share_kv_projection",
          second_read_type="from_scratch",
          summary_postprocessing_type="none",
          share_kv_projections=False),
      dict(
          testcase_name="enable_cross_block_attention",
          second_read_type="from_scratch",
          summary_postprocessing_type="none",
          cross_block_attention_mode="batch"),
      dict(
          testcase_name="new_layers",
          second_read_type="new_layers",
          summary_postprocessing_type="linear",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=False),
      dict(
          testcase_name="summary_postprocessing_transformer",
          second_read_type="from_scratch",
          summary_postprocessing_type="transformer",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=False),
      dict(
          testcase_name="new_layers_summary_postprocessing_transformer",
          second_read_type="new_layers",
          summary_postprocessing_type="transformer",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=False),
      dict(
          testcase_name="new_layers_summary_postprocessing_transformer_2",
          second_read_type="new_layers",
          summary_postprocessing_type="transformer",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=False),
      dict(
          testcase_name="cross_attend_once",
          second_read_type="cross_attend_once",
          summary_postprocessing_type="none",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=False),
      dict(
          testcase_name="cross_attend_once_2",
          second_read_type="cross_attend_once",
          summary_postprocessing_type="none",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=True),
      dict(
          testcase_name="cross_attend_once_3",
          second_read_type="cross_attend_once",
          summary_postprocessing_type="transformer",
          second_read_num_cross_attention_heads=0,
          second_read_enable_default_side_input=True),
      dict(
          testcase_name="cross_attend_once_with_absolute_add_ln_pos_emb_1",
          second_read_type="cross_attend_once",
          summary_postprocessing_type="none",
          second_read_num_cross_attention_heads=0,
          second_read_enable_default_side_input=True,
          cross_attention_pos_emb_mode="absolute_add_ln"),
      dict(
          testcase_name="cross_attend_once_with_absolute_pos_emb_1",
          second_read_type="cross_attend_once",
          summary_postprocessing_type="none",
          second_read_num_cross_attention_heads=0,
          second_read_enable_default_side_input=True,
          cross_attention_pos_emb_mode="absolute"),
      dict(
          testcase_name="cross_attend_once_with_simple_relative_pos_emb_2",
          second_read_type="cross_attend_once",
          summary_postprocessing_type="linear",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=True,
          cross_attention_pos_emb_mode="simple_relative"),
      dict(
          testcase_name="cross_attend_once_with_query_dot_relative_pos_emb_3",
          second_read_type="cross_attend_once",
          summary_postprocessing_type="linear",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=False,
          cross_attention_pos_emb_mode="query_dot_relative"),
      dict(
          testcase_name="cross_attend_once_with_absolute_pos_emb_4",
          second_read_type="cross_attend_once",
          summary_postprocessing_type="none",
          second_read_num_cross_attention_heads=0,
          second_read_enable_default_side_input=False,
          cross_attention_pos_emb_mode="absolute"),
      dict(
          testcase_name="new_layers_cross_attention",
          second_read_type="new_layers_cross_attention",
          summary_postprocessing_type="linear",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=False),
      dict(
          testcase_name="new_layers_cross_attention_2",
          second_read_type="new_layers_cross_attention",
          summary_postprocessing_type="none",
          second_read_num_cross_attention_heads=2,
          second_read_enable_default_side_input=True),
      dict(
          testcase_name="new_layers_cross_attention_3",
          second_read_type="new_layers_cross_attention",
          summary_postprocessing_type="pos",
          second_read_num_cross_attention_heads=0,
          second_read_enable_default_side_input=True),
  )
  def test_model(self,
                 second_read_type,
                 summary_postprocessing_type,
                 second_read_num_cross_attention_heads=None,
                 second_read_enable_default_side_input=False,
                 use_att_mask=True,
                 share_kv_projections=True,
                 cross_block_attention_mode="doc",
                 cross_attention_pos_emb_mode=None):
    np.random.seed(31415)
    batch_size = 7
    seq_length = 4
    max_seq_length = 19
    hidden_size = 12
    intermediate_size = 13
    num_hidden_layers = 2
    num_attention_heads = 3
    vocab_size = 50
    num_annotations = 4
    max_annotation_length = 5
    use_one_hot_embeddings = False

    second_read_num_new_layers = (
        2 if summary_postprocessing_type != "from_scratch" else None)
    summary_postprocessing_num_layers = (
        2 if summary_postprocessing_type == "transformer" else None)

    block_ids_np = np.random.randint(
        batch_size + 1, size=[batch_size], dtype=np.int32)
    block_pos_np = np.random.randint(
        batch_size + 1, size=[batch_size], dtype=np.int32)

    annotation_labels = np.random.randint(
        1e9, size=[batch_size, num_annotations], dtype=np.int32)
    annotation_labels *= np.random.binomial(
        1, 0.5, size=[batch_size, num_annotations])
    annotation_begins = np.random.randint(
        seq_length, size=[batch_size, num_annotations], dtype=np.int32)
    annotation_length = np.random.randint(
        max_annotation_length,
        size=[batch_size, num_annotations],
        dtype=np.int32)
    annotation_ends = np.minimum(annotation_begins + annotation_length,
                                 seq_length - 1)

    for summary_mode in ["cls", "text_block", "entity"]:
      if summary_mode == "text_block":
        text_block_extract_every_x = 2
      else:
        text_block_extract_every_x = None
      for use_sparse_memory_attention in [False, True]:
        config = model_config.ReadItTwiceBertConfig(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            cross_attention_pos_emb_mode=cross_attention_pos_emb_mode,
            hidden_size=hidden_size,
            share_kv_projections=share_kv_projections,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            second_read_type=second_read_type,
            second_read_num_new_layers=second_read_num_new_layers,
            second_read_num_cross_attention_heads=second_read_num_cross_attention_heads,
            second_read_enable_default_side_input=second_read_enable_default_side_input,
            summary_mode=summary_mode,
            summary_postprocessing_type=summary_postprocessing_type,
            summary_postprocessing_num_layers=summary_postprocessing_num_layers,
            use_sparse_memory_attention=use_sparse_memory_attention,
            text_block_extract_every_x=text_block_extract_every_x)

        for enable_side_inputs in [False, True]:
          for training in [False, True]:
            model = modeling.ReadItTwiceBertModel(
                config=config, use_one_hot_embeddings=use_one_hot_embeddings)

            input_ids = tf.compat.v1.placeholder_with_default(
                np.random.randint(
                    vocab_size, size=[batch_size, seq_length], dtype=np.int32),
                shape=[None, None])
            block_ids = tf.compat.v1.placeholder_with_default(
                block_ids_np, shape=[None])
            block_pos = tf.compat.v1.placeholder_with_default(
                block_pos_np, shape=[None])

            if summary_mode == "entity" or bool(np.random.binomial(1, 0.5)):
              annotation_labels_tf = tf.compat.v1.placeholder_with_default(
                  annotation_labels, shape=[None, None])
              annotation_begins_tf = tf.compat.v1.placeholder_with_default(
                  annotation_begins, shape=[None, None])
              annotation_ends_tf = tf.compat.v1.placeholder_with_default(
                  annotation_ends, shape=[None, None])
            else:
              annotation_begins_tf = None
              annotation_ends_tf = None
              annotation_labels_tf = None

            if use_att_mask:
              att_mask = tf.compat.v1.placeholder_with_default(
                  np.random.randint(
                      2,
                      size=[batch_size, seq_length, seq_length],
                      dtype=np.int32),
                  shape=[None, None, None])
            else:
              att_mask = None

            output = model(
                token_ids=input_ids,
                training=training,
                block_ids=block_ids,
                block_pos=block_pos,
                att_mask=att_mask,
                annotation_begins=annotation_begins_tf,
                annotation_ends=annotation_ends_tf,
                annotation_labels=annotation_labels_tf,
                enable_side_inputs=enable_side_inputs,
                cross_block_attention_mode=cross_block_attention_mode,
            )

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            self.evaluate(init_op)

            (final_hidden_states, local_summary_states, local_block_ids,
             local_block_pos, global_summary_states, processed_summary_states,
             global_block_ids, global_block_pos) = self.evaluate(
                 tf.tuple((
                     output.final_hidden_states,
                     output.local_summary.states,
                     output.local_summary.block_ids,
                     output.local_summary.block_pos,
                     output.global_summary.states,
                     output.global_summary.processed_states,
                     output.global_summary.block_ids,
                     output.global_summary.block_pos,
                 )))

            self.assertAllEqual(final_hidden_states.shape,
                                [batch_size, seq_length, config.hidden_size])
            if summary_mode == "cls" and not enable_side_inputs:
              self.assertNDArrayNear(final_hidden_states[:, 0, :],
                                     local_summary_states, 1e-4)

            self.assertAllEqual(local_summary_states, global_summary_states)
            if summary_mode == "cls":
              if summary_postprocessing_type == "none":
                self.assertAllEqual(local_summary_states,
                                    processed_summary_states)

                self.assertAllEqual(local_block_ids, block_ids_np)
                self.assertAllEqual(global_block_ids, block_ids_np)
                self.assertAllEqual(local_block_pos, block_pos_np)
                self.assertAllEqual(global_block_pos, block_pos_np)

            tf.reset_default_graph()
            self._ClearCachedSession()


class SummaryExtractionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="none_1",
          block_ids=[1, 1, 1],
          postprocessing_type="none",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="none_2",
          block_ids=[1, 0, 1, 2, 3],
          postprocessing_type="none",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="linear_1",
          block_ids=[1, 1, 1, 0],
          postprocessing_type="linear",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="linear_2",
          block_ids=[1, 1, 2, 3],
          postprocessing_type="linear",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="pos_1",
          block_ids=[1, 0, 1, 1],
          postprocessing_type="pos",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="pos_2",
          block_ids=[1, 1, 2, 3],
          postprocessing_type="pos",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="transformer_1",
          block_ids=[1, 1, 1, 0],
          postprocessing_type="transformer",
          postprocessing_num_layers=2,
      ),
      dict(
          testcase_name="transformer_2",
          block_ids=[1, 1, 2, 3, 0],
          postprocessing_type="transformer",
          postprocessing_num_layers=2,
      ),
  )
  def test_cls_summary_extraction(self, block_ids, postprocessing_type,
                                  postprocessing_num_layers):

    hidden_size = 8
    seq_length = 7
    block_ids_np = np.array(block_ids)
    batch_size = len(block_ids)
    block_pos_np = np.random.randint(
        batch_size, size=[batch_size], dtype=np.int32)
    cross_block_attention_mode = "doc"
    hidden_states_np = np.random.random(
        (batch_size, seq_length, hidden_size)).astype(np.float32)

    for use_one_hot_embeddings in [False, True]:
      for training in [False, True]:
        block_ids_tf = tf.compat.v1.placeholder_with_default(
            block_ids_np, shape=[None])
        block_pos_tf = tf.compat.v1.placeholder_with_default(
            block_pos_np, shape=[None])
        hidden_states_tf = tf.compat.v1.placeholder_with_default(
            hidden_states_np, shape=[None, None, hidden_size])

        config = model_config.ReadItTwiceBertConfig(
            vocab_size=40,
            max_seq_length=40,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 3,
            num_hidden_layers=2,
            num_attention_heads=2,
            second_read_type="from_scratch",
            second_read_num_new_layers=None,
            second_read_num_cross_attention_heads=None,
            second_read_enable_default_side_input=None,
            summary_mode="cls",
            summary_postprocessing_type=postprocessing_type,
            summary_postprocessing_num_layers=postprocessing_num_layers,
            use_sparse_memory_attention=False)

        summary_extraction = modeling.SummaryExtraction(
            config=config, use_one_hot_embeddings=use_one_hot_embeddings)

        output = summary_extraction(
            hidden_states_tf,
            block_ids_tf,
            block_pos_tf,
            annotation_begins=None,
            annotation_ends=None,
            annotation_labels=None,
            main_seq_length=tf.compat.v1.placeholder_with_default(
                seq_length, shape=[]),
            num_replicas_concat=None,
            cross_block_attention_mode=cross_block_attention_mode,
            training=training)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.evaluate(init_op)

        (local_summary_states, local_block_ids, local_block_pos, local_labels,
         global_summary_states, processed_summary_states, global_block_ids,
         global_block_pos, global_labels,
         token_to_global_summary_att_map) = self.evaluate(
             tf.tuple((
                 output.local_summary.states,
                 output.local_summary.block_ids,
                 output.local_summary.block_pos,
                 output.local_summary.labels,
                 output.global_summary.states,
                 output.global_summary.processed_states,
                 output.global_summary.block_ids,
                 output.global_summary.block_pos,
                 output.local_summary.labels,
                 output.token_to_global_summary_att_map,
             )))

        self.assertAllEqual(local_summary_states, global_summary_states)
        if postprocessing_type == "none":
          self.assertAllEqual(local_summary_states, processed_summary_states)

        self.assertAllEqual(hidden_states_np[:, 0], local_summary_states)

        self.assertAllEqual(local_block_ids, block_ids_np)
        self.assertAllEqual(global_block_ids, block_ids_np)
        self.assertAllEqual(local_block_pos, block_pos_np)
        self.assertAllEqual(global_block_pos, block_pos_np)
        self.assertAllEqual(local_labels, block_ids_np)
        self.assertAllEqual(global_labels, block_ids_np)

        self.assertAllEqual([batch_size, seq_length, batch_size],
                            token_to_global_summary_att_map.shape)

        for i in range(batch_size):
          for j in range(batch_size):
            x = int((block_ids_np[i] != 0) and (block_ids_np[j] != 0) and
                    (block_ids_np[i] == block_ids_np[j]))
            self.assertAllEqual([x] * seq_length,
                                token_to_global_summary_att_map[i, :, j])

        tf.reset_default_graph()
        self._ClearCachedSession()

  @parameterized.named_parameters(
      dict(
          testcase_name="none_1",
          block_ids=[1, 1, 1],
          postprocessing_type="none",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="none_2",
          block_ids=[1, 0, 1, 2, 3],
          postprocessing_type="none",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="linear_1",
          block_ids=[1, 1, 1, 0],
          postprocessing_type="linear",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="linear_2",
          block_ids=[1, 1, 2, 3],
          postprocessing_type="linear",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="pos_1",
          block_ids=[1, 0, 1, 1],
          postprocessing_type="pos",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="pos_2",
          block_ids=[1, 1, 2, 3],
          postprocessing_type="pos",
          postprocessing_num_layers=None,
      ),
      dict(
          testcase_name="transformer_1",
          block_ids=[1, 1, 1, 0],
          postprocessing_type="transformer",
          postprocessing_num_layers=2,
      ),
      dict(
          testcase_name="transformer_2",
          block_ids=[1, 1, 2, 3, 0],
          postprocessing_type="transformer",
          postprocessing_num_layers=2,
      ),
  )
  def test_text_block_summary_extraction(self, block_ids, postprocessing_type,
                                         postprocessing_num_layers):

    hidden_size = 8
    seq_length = 9
    block_ids_np = np.array(block_ids)
    batch_size = len(block_ids)
    block_pos_np = np.random.randint(
        batch_size, size=[batch_size], dtype=np.int32)
    cross_block_attention_mode = "doc"
    text_block_extract_every_x = 3
    assert seq_length % text_block_extract_every_x == 0
    num_annotations = seq_length // text_block_extract_every_x

    hidden_states_np = np.random.random(
        (batch_size, seq_length, hidden_size)).astype(np.float32)
    token_ids_np = np.random.randint(
        batch_size, size=[batch_size, seq_length], dtype=np.int32)

    for use_one_hot_embeddings in [False, True]:
      for training in [False, True]:
        block_ids_tf = tf.compat.v1.placeholder_with_default(
            block_ids_np, shape=[None])
        block_pos_tf = tf.compat.v1.placeholder_with_default(
            block_pos_np, shape=[None])
        hidden_states_tf = tf.compat.v1.placeholder_with_default(
            hidden_states_np, shape=[None, None, hidden_size])
        token_ids_tf = tf.compat.v1.placeholder_with_default(
            token_ids_np, shape=[None, None])
        config = model_config.ReadItTwiceBertConfig(
            vocab_size=40,
            max_seq_length=40,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 3,
            num_hidden_layers=2,
            num_attention_heads=2,
            second_read_type="from_scratch",
            second_read_num_new_layers=None,
            second_read_num_cross_attention_heads=None,
            second_read_enable_default_side_input=None,
            summary_mode="text_block",
            summary_postprocessing_type=postprocessing_type,
            summary_postprocessing_num_layers=postprocessing_num_layers,
            text_block_extract_every_x=text_block_extract_every_x,
            use_sparse_memory_attention=False)

        summary_extraction = modeling.SummaryExtraction(
            config=config, use_one_hot_embeddings=use_one_hot_embeddings)

        output = summary_extraction(
            hidden_states_tf,
            block_ids_tf,
            block_pos_tf,
            annotation_begins=None,
            annotation_ends=None,
            annotation_labels=None,
            main_seq_length=tf.compat.v1.placeholder_with_default(
                seq_length, shape=[]),
            num_replicas_concat=None,
            cross_block_attention_mode=cross_block_attention_mode,
            training=training,
            token_ids=token_ids_tf)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.evaluate(init_op)

        (local_summary_states, local_block_ids, local_block_pos, local_labels,
         global_summary_states, processed_summary_states, global_block_ids,
         global_block_pos, global_labels) = self.evaluate(
             tf.tuple((
                 output.local_summary.states,
                 output.local_summary.block_ids,
                 output.local_summary.block_pos,
                 output.local_summary.labels,
                 output.global_summary.states,
                 output.global_summary.processed_states,
                 output.global_summary.block_ids,
                 output.global_summary.block_pos,
                 output.local_summary.labels,
             )))
        tf.reset_default_graph()
        self._ClearCachedSession()

        self.assertAllEqual(local_summary_states, global_summary_states)
        if postprocessing_type == "none":
          self.assertAllEqual(local_summary_states, processed_summary_states)

        def tile(x):
          return np.tile(np.expand_dims(x, 1), [1, num_annotations]).reshape(-1)

        self.assertAllEqual(local_block_ids, tile(block_ids_np))
        self.assertAllEqual(global_block_ids, tile(block_ids_np))
        self.assertAllEqual(local_block_pos, tile(block_pos_np))
        self.assertAllEqual(global_block_pos, tile(block_pos_np))
        self.assertAllEqual(local_labels, global_labels)

        local_labels = local_labels.reshape(batch_size, num_annotations)
        for i in range(batch_size):
          k = 0
          for l in range(0, seq_length, text_block_extract_every_x):
            is_pad_token = ((token_ids_np[i, l] == 0) or
                            (token_ids_np[i, l + text_block_extract_every_x -
                                          1] == 0))
            self.assertEqual(local_labels[i, k], int(not is_pad_token))
            k += 1

  @parameterized.named_parameters(
      dict(
          testcase_name="none_1",
          block_ids=[1, 1, 1],
          postprocessing_type="none",
          postprocessing_num_layers=None,
          num_annotations=1,
          max_annotation_length=1,
      ),
      dict(
          testcase_name="none_2",
          block_ids=[1, 1, 1],
          postprocessing_type="none",
          postprocessing_num_layers=None,
          num_annotations=10,
          max_annotation_length=3,
      ),
      dict(
          testcase_name="none_3",
          block_ids=[1, 1, 2, 3],
          postprocessing_type="none",
          postprocessing_num_layers=None,
          num_annotations=2,
          max_annotation_length=3,
      ),
      dict(
          testcase_name="linear_1",
          block_ids=[1, 1, 1],
          postprocessing_type="linear",
          postprocessing_num_layers=None,
          num_annotations=13,
          max_annotation_length=2,
      ),
      dict(
          testcase_name="linear_2",
          block_ids=[1, 1, 2, 3],
          postprocessing_type="linear",
          postprocessing_num_layers=None,
          num_annotations=13,
          max_annotation_length=100,
      ),
      dict(
          testcase_name="pos_1",
          block_ids=[1, 1, 1],
          postprocessing_type="pos",
          postprocessing_num_layers=None,
          num_annotations=7,
          max_annotation_length=1,
      ),
      dict(
          testcase_name="pos_2",
          block_ids=[1, 1, 2, 3],
          postprocessing_type="pos",
          postprocessing_num_layers=None,
          num_annotations=23,
          max_annotation_length=5,
      ),
      dict(
          testcase_name="transformer_1",
          block_ids=[1, 1, 1],
          postprocessing_type="transformer",
          postprocessing_num_layers=2,
          num_annotations=4,
          max_annotation_length=5,
      ),
      dict(
          testcase_name="transformer_2",
          block_ids=[1, 1, 2, 3],
          postprocessing_type="transformer",
          postprocessing_num_layers=2,
          num_annotations=5,
          max_annotation_length=6,
      ),
  )
  def test_entity_summary_extraction(self, block_ids, postprocessing_type,
                                     postprocessing_num_layers, num_annotations,
                                     max_annotation_length):
    hidden_size = 8
    seq_length = 7
    block_ids_np = np.array(block_ids)
    batch_size = len(block_ids)
    block_pos_np = np.random.randint(
        batch_size, size=[batch_size], dtype=np.int32)
    cross_block_attention_mode = "doc"
    hidden_states_np = np.random.random(
        (batch_size, seq_length, hidden_size)).astype(np.float32)
    annotation_labels = np.random.randint(
        1e9, size=[batch_size, num_annotations], dtype=np.int32)
    annotation_labels *= np.random.binomial(
        1, 0.5, size=[batch_size, num_annotations])
    annotation_begins = np.random.randint(
        seq_length, size=[batch_size, num_annotations], dtype=np.int32)
    annotation_length = np.random.randint(
        max_annotation_length,
        size=[batch_size, num_annotations],
        dtype=np.int32)
    annotation_ends = np.minimum(annotation_begins + annotation_length,
                                 seq_length - 1)

    for use_one_hot_embeddings in [False, True]:
      for training in [False, True]:
        block_ids_tf = tf.compat.v1.placeholder_with_default(
            block_ids_np, shape=[None])
        block_pos_tf = tf.compat.v1.placeholder_with_default(
            block_pos_np, shape=[None])
        hidden_states_tf = tf.compat.v1.placeholder_with_default(
            hidden_states_np, shape=[None, None, hidden_size])
        annotation_labels_tf = tf.compat.v1.placeholder_with_default(
            annotation_labels, shape=[None, None])
        annotation_begins_tf = tf.compat.v1.placeholder_with_default(
            annotation_begins, shape=[None, None])
        annotation_ends_tf = tf.compat.v1.placeholder_with_default(
            annotation_ends, shape=[None, None])

        config = model_config.ReadItTwiceBertConfig(
            vocab_size=40,
            max_seq_length=40,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 3,
            num_hidden_layers=2,
            num_attention_heads=2,
            second_read_type="from_scratch",
            second_read_num_new_layers=None,
            second_read_num_cross_attention_heads=None,
            second_read_enable_default_side_input=None,
            summary_mode="entity",
            summary_postprocessing_type=postprocessing_type,
            summary_postprocessing_num_layers=postprocessing_num_layers,
            use_sparse_memory_attention=True)

        summary_extraction = modeling.SummaryExtraction(
            config=config, use_one_hot_embeddings=use_one_hot_embeddings)

        output = summary_extraction(
            hidden_states_tf,
            block_ids_tf,
            block_pos_tf,
            annotation_begins=annotation_begins_tf,
            annotation_ends=annotation_ends_tf,
            annotation_labels=annotation_labels_tf,
            main_seq_length=tf.compat.v1.placeholder_with_default(
                seq_length, shape=[]),
            num_replicas_concat=None,
            cross_block_attention_mode=cross_block_attention_mode,
            training=training)

        # init_op = tf.group(tf.global_variables_initializer(),
        #                    tf.local_variables_initializer())
        # self.evaluate(init_op)
        self.evaluate(tf.local_variables_initializer())
        self.evaluate(tf.global_variables_initializer())

        weight_np = self.evaluate(summary_extraction.extraction_linear.kernel)

        (local_summary_states, local_block_ids, local_block_pos, local_labels,
         global_summary_states, processed_summary_states, global_block_ids,
         global_block_pos, global_labels,
         token_to_global_summary_att_map) = self.evaluate(
             tf.tuple((
                 output.local_summary.states,
                 output.local_summary.block_ids,
                 output.local_summary.block_pos,
                 output.local_summary.labels,
                 output.global_summary.states,
                 output.global_summary.processed_states,
                 output.global_summary.block_ids,
                 output.global_summary.block_pos,
                 output.local_summary.labels,
                 output.token_to_global_summary_att_map,
             )))

        self.assertAllEqual(local_summary_states, global_summary_states)
        if postprocessing_type == "none":
          self.assertAllEqual(local_summary_states, processed_summary_states)

        self.assertAllEqual([batch_size * num_annotations, hidden_size],
                            local_summary_states.shape)

        k = 0
        for i in range(batch_size):
          for j in range(num_annotations):
            begin_state = hidden_states_np[i, annotation_begins[i, j]]
            end_state = hidden_states_np[i, annotation_ends[i, j]]
            summary_np = np.dot(
                np.concatenate([begin_state, end_state]), weight_np)
            if annotation_labels[i, j] == 0:
              tol = 1e-9
              summary_np = np.zeros_like(summary_np)
            else:
              tol = 1e-5
            self.assertArrayNear(summary_np, local_summary_states[k], err=tol)
            k += 1

        def tile(x):
          return np.tile(np.expand_dims(x, 1), [1, num_annotations]).reshape(-1)

        self.assertAllEqual(local_block_ids, tile(block_ids_np))
        self.assertAllEqual(global_block_ids, tile(block_ids_np))
        self.assertAllEqual(local_block_pos, tile(block_pos_np))
        self.assertAllEqual(global_block_pos, tile(block_pos_np))
        self.assertAllEqual(local_labels, annotation_labels.reshape(-1))
        self.assertAllEqual(global_labels, annotation_labels.reshape(-1))

        self.assertAllEqual(
            [batch_size, seq_length, batch_size * num_annotations],
            token_to_global_summary_att_map.shape)

        for i in range(batch_size):
          for l in range(seq_length):
            is_token_an_entity = False
            for a in range(num_annotations):
              if (annotation_labels[i, a] != 0 and
                  annotation_begins[i, a] <= l and l <= annotation_ends[i, a]):
                is_token_an_entity = True
            k = 0
            for j1 in range(batch_size):
              for j2 in range(num_annotations):
                are_blocks_same = int((block_ids_np[i] != 0) and
                                      (global_block_ids[k] != 0) and
                                      (block_ids_np[i] == block_ids_np[j1]))
                x = int(
                    is_token_an_entity and are_blocks_same and
                    (annotation_labels[j1, j2] != 0))
                self.assertEqual(x, token_to_global_summary_att_map[i, l, k])
                k += 1

        tf.reset_default_graph()
        self._ClearCachedSession()


class SpanPredictionHeadTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="linear",
          intermediate_size=None,
          dropout_rate=0.0,
          use_token_ids=False,
          use_ignore_prefix_length=False,
          training=True),
      dict(
          testcase_name="mlp",
          intermediate_size=5,
          dropout_rate=0.0,
          use_token_ids=False,
          use_ignore_prefix_length=False,
          training=True),
      dict(
          testcase_name="mlp_dropout",
          intermediate_size=5,
          dropout_rate=0.5,
          use_token_ids=False,
          use_ignore_prefix_length=False,
          training=True),
      dict(
          testcase_name="mlp_no_training",
          intermediate_size=5,
          dropout_rate=0.0,
          use_token_ids=False,
          use_ignore_prefix_length=False,
          training=False),
      dict(
          testcase_name="mlp_use_token_ids",
          intermediate_size=5,
          dropout_rate=0.0,
          use_token_ids=True,
          use_ignore_prefix_length=False,
          training=True),
      dict(
          testcase_name="mlp_use_ignore_prefix_length",
          intermediate_size=5,
          dropout_rate=0.0,
          use_token_ids=False,
          use_ignore_prefix_length=True,
          training=True),
      dict(
          testcase_name="mlp_use_both",
          intermediate_size=5,
          dropout_rate=0.5,
          use_token_ids=True,
          use_ignore_prefix_length=True,
          training=True),
  )
  def pan_prediction_head(self, intermediate_size, dropout_rate, use_token_ids,
                          use_ignore_prefix_length, training):
    np.random.seed(31415)
    batch_size = 3
    seq_length = 7
    hidden_size = 13

    hidden_states = tf.compat.v1.placeholder_with_default(
        np.random.random(
            (batch_size, seq_length, hidden_size)).astype(np.float32),
        shape=[None, None, hidden_size])

    if use_token_ids:
      token_ids = tf.compat.v1.placeholder_with_default(
          np.random.randint(2, size=[batch_size, seq_length], dtype=np.int32),
          shape=[None, None])
      padding_token_id = 0
    else:
      token_ids = None
      padding_token_id = None

    if use_ignore_prefix_length:
      ignore_prefix_length = tf.compat.v1.placeholder_with_default(
          np.random.randint(seq_length + 1, size=[batch_size], dtype=np.int32),
          shape=[None])
    else:
      ignore_prefix_length = None

    model = modeling.SpanPredictionHead(
        intermediate_size=intermediate_size, dropout_rate=dropout_rate)

    output = model(
        hidden_states=hidden_states,
        token_ids=token_ids,
        padding_token_id=padding_token_id,
        ignore_prefix_length=ignore_prefix_length,
        training=training)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.evaluate(init_op)
    output_result = self.evaluate(output)

    self.assertAllEqual(output_result.shape, [batch_size, seq_length, 2])


class ReadItTwiceDecoderModelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="num_heads_1_1",
          num_cross_attention_heads=1,
          training=True,
          use_one_hot_embeddings=True),
      dict(
          testcase_name="num_heads_1_2",
          num_cross_attention_heads=1,
          training=True,
          use_one_hot_embeddings=False),
      dict(
          testcase_name="num_heads_1_3",
          num_cross_attention_heads=1,
          training=False,
          use_one_hot_embeddings=True),
      dict(
          testcase_name="num_heads_1_4",
          num_cross_attention_heads=1,
          training=False,
          use_one_hot_embeddings=False),
      dict(
          testcase_name="num_heads_3_1",
          num_cross_attention_heads=3,
          training=True,
          use_one_hot_embeddings=True),
      dict(
          testcase_name="num_heads_3_2",
          num_cross_attention_heads=3,
          training=True,
          use_one_hot_embeddings=False),
      dict(
          testcase_name="num_heads_3_3",
          num_cross_attention_heads=3,
          training=False,
          use_one_hot_embeddings=True),
      dict(
          testcase_name="num_heads_3_4",
          num_cross_attention_heads=3,
          training=False,
          use_one_hot_embeddings=False),
      dict(
          testcase_name="num_heads_0_1",
          num_cross_attention_heads=0,
          training=True,
          use_one_hot_embeddings=True),
      dict(
          testcase_name="num_heads_0_2",
          num_cross_attention_heads=0,
          training=True,
          use_one_hot_embeddings=False),
      dict(
          testcase_name="num_heads_0_3",
          num_cross_attention_heads=0,
          training=False,
          use_one_hot_embeddings=True),
      dict(
          testcase_name="num_heads_0_4",
          num_cross_attention_heads=0,
          training=False,
          use_one_hot_embeddings=False),
  )
  def test_model(self, num_cross_attention_heads, training,
                 use_one_hot_embeddings):
    batch_size = 3
    seq_length = 7
    hidden_size = 12
    side_seq_length = 6
    seq_length = 11
    num_hidden_layers = 3

    config = model_config.ReadItTwiceBertConfig(
        vocab_size=100,
        max_seq_length=30,
        hidden_size=hidden_size,
        intermediate_size=36,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        use_sparse_memory_attention=False)

    np.random.seed(31415)
    input_ids = tf.compat.v1.placeholder_with_default(
        np.random.randint(
            config.vocab_size, size=[batch_size, seq_length], dtype=np.int32),
        shape=[None, None])

    side_input = tf.compat.v1.placeholder_with_default(
        np.random.random((side_seq_length, hidden_size)).astype(np.float32),
        shape=[None, hidden_size])

    att_mask = tf.compat.v1.placeholder_with_default(
        np.random.randint(
            2, size=[batch_size, side_seq_length], dtype=np.int32),
        shape=[None, None])

    model = modeling.ReadItTwiceDecoderModel(
        config=config,
        num_layers_override=num_hidden_layers,
        num_cross_attention_heads=num_cross_attention_heads,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output = model(
        token_ids=input_ids,
        side_input=side_input,
        token2side_input_att_mask=att_mask,
        training=training,
    )

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.evaluate(init_op)
    output_result = self.evaluate(output)

    self.assertAllEqual(output_result.shape,
                        [batch_size, seq_length, config.hidden_size])

  @parameterized.named_parameters(
      dict(
          testcase_name="single_block_1_doc",
          block_ids=[1, 3],
          global_block_ids=[4, 1, 3],
          cross_block_attention_mode="doc",
          expected_attn=[[0, 1, 0], [0, 0, 1]]),
      dict(
          testcase_name="single_block_1_batch",
          block_ids=[1, 3],
          global_block_ids=[4, 1, 3],
          cross_block_attention_mode="batch",
          expected_attn=[[1, 1, 1], [1, 1, 1]]),
      dict(
          testcase_name="single_block_1_block",
          block_ids=[1, 3],
          block_pos=[1, 1],
          global_block_ids=[4, 1, 3],
          global_block_pos=[1, 1, 2],
          cross_block_attention_mode="block",
          expected_attn=[[0, 1, 0], [0, 0, 0]]),
      dict(
          testcase_name="single_block_1_other_blocks",
          block_ids=[1, 3],
          block_pos=[1, 1],
          global_block_ids=[4, 1, 3],
          global_block_pos=[1, 1, 2],
          cross_block_attention_mode="other_blocks",
          expected_attn=[[0, 0, 0], [0, 0, 1]]),
      dict(
          testcase_name="single_block_2_doc",
          block_ids=[1, 3, 0],
          global_block_ids=[4, 1, 3, 0],
          cross_block_attention_mode="doc",
          expected_attn=[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
      dict(
          testcase_name="single_block_2_batch",
          block_ids=[1, 3, 0],
          global_block_ids=[4, 1, 3, 0],
          cross_block_attention_mode="batch",
          expected_attn=[[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]),
      dict(
          testcase_name="single_block_2_block",
          block_ids=[1, 3, 0],
          block_pos=[239, 300, 100],
          global_block_ids=[4, 1, 3, 0],
          global_block_pos=[200, 100, 300, -1],
          cross_block_attention_mode="block",
          expected_attn=[[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
      dict(
          testcase_name="single_block_2_other_blocks",
          block_ids=[1, 3, 0],
          block_pos=[239, 300, 100],
          global_block_ids=[4, 1, 3, 0],
          global_block_pos=[200, 100, 300, -1],
          cross_block_attention_mode="other_blocks",
          expected_attn=[[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
      dict(
          testcase_name="globals_all_zeros",
          block_ids=[1, 3],
          global_block_ids=[0, 0, 0],
          cross_block_attention_mode="doc",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="globals_all_zeros_batch",
          block_ids=[1, 3],
          global_block_ids=[0, 0, 0],
          cross_block_attention_mode="batch",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="globals_all_zeros_block",
          block_ids=[1, 3],
          global_block_ids=[0, 0, 0],
          cross_block_attention_mode="block",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="locals_all_zeros",
          block_ids=[0, 0],
          global_block_ids=[4, 1, 3],
          cross_block_attention_mode="doc",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="locals_all_zeros_batch",
          block_ids=[0, 0],
          global_block_ids=[4, 1, 3],
          cross_block_attention_mode="batch",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="locals_all_zeros_block",
          block_ids=[0, 0],
          global_block_ids=[4, 1, 3],
          cross_block_attention_mode="block",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="all_zeros",
          block_ids=[0, 0],
          global_block_ids=[0, 0, 0],
          cross_block_attention_mode="doc",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="all_zeros_batch",
          block_ids=[0, 0],
          global_block_ids=[0, 0, 0],
          cross_block_attention_mode="batch",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="all_zeros_block",
          block_ids=[0, 0],
          global_block_ids=[0, 0, 0],
          cross_block_attention_mode="block",
          expected_attn=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="multi_block_1",
          block_ids=[1, 3, 1],
          global_block_ids=[4, 1, 3, 1],
          cross_block_attention_mode="doc",
          expected_attn=[[0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]]),
      dict(
          testcase_name="multi_block_1_block",
          block_ids=[1, 3, 1],
          block_pos=[1, 1, 2],
          global_block_ids=[4, 1, 3, 1],
          global_block_pos=[1, 1, 1, 2],
          cross_block_attention_mode="block",
          expected_attn=[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
      dict(
          testcase_name="multi_block_1_other_blocks",
          block_ids=[1, 3, 1],
          block_pos=[1, 1, 2],
          global_block_ids=[4, 1, 3, 1],
          global_block_pos=[1, 1, 1, 2],
          cross_block_attention_mode="other_blocks",
          expected_attn=[[0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]]),
      dict(
          testcase_name="multi_block_2",
          block_ids=[1, 3, 1, 3],
          global_block_ids=[1, 3, 1],
          cross_block_attention_mode="doc",
          expected_attn=[[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]]),
      dict(
          testcase_name="multi_block_2_block",
          block_ids=[1, 3, 1, 3],
          block_pos=[2, 2, 1, 1],
          global_block_ids=[1, 3, 1],
          global_block_pos=[2, 1, 2],
          cross_block_attention_mode="block",
          expected_attn=[[1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0]]),
  )
  def test_get_cross_block_att(self,
                               block_ids,
                               global_block_ids,
                               cross_block_attention_mode,
                               expected_attn,
                               block_pos=None,
                               global_block_pos=None):
    block_ids_tf = tf.compat.v1.placeholder_with_default(
        np.array(block_ids).astype(np.int32), shape=[None])
    global_block_ids_tf = tf.compat.v1.placeholder_with_default(
        np.array(global_block_ids).astype(np.int32), shape=[None])
    block_pos_tf = tf.compat.v1.placeholder_with_default(
        np.array(block_pos or np.arange(len(block_ids))).astype(np.int32),
        shape=[None])
    global_block_pos_tf = tf.compat.v1.placeholder_with_default(
        np.array(global_block_pos or
                 np.arange(len(global_block_ids))).astype(np.int32),
        shape=[None])

    actual_attn = self.evaluate(
        modeling.get_cross_block_att(block_ids_tf, block_pos_tf,
                                     global_block_ids_tf, global_block_pos_tf,
                                     cross_block_attention_mode))
    self.assertAllEqual(expected_attn, actual_attn)


if __name__ == "__main__":
  tf.test.main()
