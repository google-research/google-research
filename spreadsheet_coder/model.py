# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Build model essentials."""

import numpy as np
from tensor2tensor.utils import beam_search
import tensorflow.compat.v1 as tf
import tensorflow.contrib.rnn as contrib_rnn
import tf_slim as slim

from spreadsheet_coder import bert_modeling
from spreadsheet_coder import constants
from spreadsheet_coder import mobilebert_modeling
from spreadsheet_coder import model_utils


def create_model(
    num_encoder_layers, num_decoder_layers, embedding_size, hidden_size,
    dropout_rate, is_training, formula, row_cell_context, row_context_mask,
    row_context_segment_ids, row_cell_indices, row_context_mask_per_cell,
    row_context_segment_ids_per_cell, col_cell_context, col_context_mask,
    col_context_segment_ids, col_cell_indices, col_context_mask_per_cell,
    col_context_segment_ids_per_cell, exclude_headers, max_cell_context_length,
    num_rows, record_index, column_index, layer_norm, cell_position_encoding,
    cell_context_encoding, use_bert, use_mobilebert, per_row_encoding,
    max_pooling, use_cnn, use_pointer_network, two_stage_decoding, conv_type,
    grid_type, skip_connection, bert_config, unused_tensors_to_print,
    formula_length, formula_prefix_length, vocab_size, beam_size, use_tpu,
    use_one_hot_embeddings):
  """Creates a program generator."""
  use_dropout = is_training
  input_shape = bert_modeling.get_shape_list(
      formula, expected_rank=2)
  batch_size = input_shape[0]

  height = 22
  row_width = 21
  col_width = 22

  if exclude_headers:
    row_context_mask *= tf.cast(row_context_segment_ids, dtype=tf.float32)
    row_context_mask_per_cell *= tf.cast(row_context_segment_ids_per_cell,
                                         dtype=tf.float32)
    col_context_mask *= tf.cast(col_context_segment_ids, dtype=tf.float32)
    col_context_mask_per_cell *= tf.cast(col_context_segment_ids_per_cell,
                                         dtype=tf.float32)
    row_cell_context *= row_context_segment_ids

  if num_rows < 21:
    cell_data_mask = ([1] * max_cell_context_length +
                      [0] * max_cell_context_length * (10 - num_rows) +
                      [1] * max_cell_context_length * (num_rows + 1) +
                      [0] * max_cell_context_length * 10)
    cell_data_mask = tf.convert_to_tensor(np.array(cell_data_mask),
                                          dtype=tf.float32)
    cell_data_mask = tf.expand_dims(cell_data_mask, dim=0)
    cell_data_mask_per_cell = ([1] * 21 + [0] * 21 * (10 - num_rows) +
                               [1] * 21 * (num_rows + 1) + [0] * 21 * 10)
    cell_data_mask_per_cell = tf.convert_to_tensor(
        np.array(cell_data_mask_per_cell), dtype=tf.float32)
    cell_data_mask_per_cell = tf.expand_dims(cell_data_mask_per_cell, dim=0)
    row_cell_context *= tf.cast(cell_data_mask, dtype=tf.int32)
    row_context_mask *= cell_data_mask
    row_context_mask_per_cell *= cell_data_mask_per_cell

  if cell_context_encoding:
    if grid_type != "col":
      reshape_row_cell_context = tf.reshape(
          row_cell_context, [batch_size, height, max_cell_context_length])
      reshape_row_context_mask = tf.reshape(
          row_context_mask, [batch_size, height, max_cell_context_length])
      reshape_row_context_segment_ids = tf.reshape(row_context_segment_ids,
                                                   [batch_size, height,
                                                    max_cell_context_length])
      split_row_cell_context = tf.split(reshape_row_cell_context, height,
                                        axis=1)
      split_row_context_mask = tf.split(reshape_row_context_mask, height,
                                        axis=1)
      split_row_context_segment_ids = tf.split(reshape_row_context_segment_ids,
                                               height, axis=1)

      if not use_bert and not use_mobilebert:
        if max_pooling:
          split_row_cell_context = ([split_row_cell_context[0]] +
                                    split_row_cell_context[2:12])
          split_row_context_mask = ([split_row_context_mask[0]] +
                                    split_row_context_mask[2:12])
          split_row_context_segment_ids = ([split_row_context_segment_ids[0]] +
                                           split_row_context_segment_ids[2:12])
          height = 11
        else:
          split_row_cell_context = split_row_cell_context[:12]
          split_row_context_mask = split_row_context_mask[:12]
          split_row_context_segment_ids = split_row_context_segment_ids[:12]
          height = 12
      header = tf.squeeze(split_row_cell_context[0], axis=1)
      header_mask = tf.squeeze(split_row_context_mask[0], axis=1)
      header_segment_ids = tf.squeeze(split_row_context_segment_ids[0], axis=1)
      row_context_grid = []
      header_encoding = []

    if grid_type != "row":
      reshape_col_cell_context = tf.reshape(
          col_cell_context, [batch_size, height, max_cell_context_length])
      reshape_col_context_mask = tf.reshape(
          col_context_mask, [batch_size, height, max_cell_context_length])
      reshape_col_context_segment_ids = tf.reshape(col_context_segment_ids,
                                                   [batch_size, height,
                                                    max_cell_context_length])
      split_col_cell_context = tf.split(reshape_col_cell_context, height,
                                        axis=1)
      split_col_context_mask = tf.split(reshape_col_context_mask, height,
                                        axis=1)
      split_col_context_segment_ids = tf.split(reshape_col_context_segment_ids,
                                               height, axis=1)
      if not use_bert and not use_mobilebert:
        split_col_cell_context = split_col_cell_context[:12]
        split_col_context_mask = split_col_context_mask[:12]
        split_col_context_segment_ids = split_col_context_segment_ids[:12]
        height = 12
      cur_col = tf.squeeze(split_col_cell_context[0], axis=1)
      cur_col_mask = tf.squeeze(split_col_context_mask[0], axis=1)
      cur_col_segment_ids = tf.squeeze(split_col_context_segment_ids[0], axis=1)
      col_context_grid = []

    if grid_type != "col":
      if per_row_encoding:
        chunk_size = 1
        st_idx = 0
      else:
        chunk_size = 512 // max_cell_context_length - 1
        st_idx = 1

      if use_bert or use_mobilebert:
        if grid_type == "both":
          bert_scope = "row/bert"
        else:
          bert_scope = "bert"
      else:
        bert_vocab_size = bert_config.vocab_size
        with tf.variable_scope("row", reuse=tf.AUTO_REUSE):
          row_context_embedding = tf.get_variable(
              name="cell_context_embedding",
              shape=[bert_vocab_size, embedding_size],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
          row_context_encoder_cells = [model_utils.build_lstm(hidden_size)
                                       for _ in range(num_encoder_layers)]
          row_context_encoder_cells = contrib_rnn.MultiRNNCell(
              row_context_encoder_cells, state_is_tuple=True)

      for i in range(st_idx, height, chunk_size):
        for j in range(chunk_size):
          split_row_cell_context[i + j] = tf.squeeze(
              split_row_cell_context[i + j], axis=1)
          split_row_context_mask[i + j] = tf.squeeze(
              split_row_context_mask[i + j], axis=1)
          split_row_context_segment_ids[i + j] = tf.squeeze(
              split_row_context_segment_ids[i + j], axis=1)
        if per_row_encoding:
          concat_row_cell_context = split_row_cell_context[i]
          concat_row_mask = split_row_context_mask[i]
          concat_row_segment_ids = split_row_context_segment_ids[i]
        else:
          concat_row_cell_context = tf.concat(
              [header] + split_row_cell_context[i: i + chunk_size], axis=-1)
          concat_row_mask = tf.concat(
              [header_mask] + split_row_context_mask[i: i + chunk_size],
              axis=-1)
          concat_row_segment_ids = tf.concat(
              [header_segment_ids] +
              split_row_context_segment_ids[i:i + chunk_size],
              axis=-1)
        if use_mobilebert:
          row_bert_context_model = mobilebert_modeling.BertModel(
              config=bert_config, is_training=is_training,
              input_ids=concat_row_cell_context, input_mask=concat_row_mask,
              token_type_ids=concat_row_segment_ids,
              use_one_hot_embeddings=use_one_hot_embeddings,
              scope=bert_scope)
        elif use_bert:
          row_bert_context_model = bert_modeling.BertModel(
              config=bert_config, is_training=is_training,
              input_ids=concat_row_cell_context, input_mask=concat_row_mask,
              token_type_ids=concat_row_segment_ids,
              use_one_hot_embeddings=use_one_hot_embeddings,
              scope=bert_scope)
        else:
          cell_context_embeddings = tf.nn.embedding_lookup(
              row_context_embedding, concat_row_cell_context)
          row_context_sequence_output, _ = tf.nn.dynamic_rnn(
              row_context_encoder_cells,
              cell_context_embeddings,
              initial_state=row_context_encoder_cells.get_initial_state(
                  batch_size=batch_size, dtype=tf.float32),
              dtype=tf.float32)
        if use_bert or use_mobilebert:
          row_context_sequence_output = (
              row_bert_context_model.get_sequence_output())
        row_context_sequence_output = tf.reshape(
            row_context_sequence_output,
            [batch_size, chunk_size + st_idx, max_cell_context_length, -1])
        row_context_sequence_output = tf.split(
            row_context_sequence_output, chunk_size + st_idx, axis=1)
        if not per_row_encoding:
          header_encoding.append(row_context_sequence_output[0])
        for j in range(st_idx, chunk_size + st_idx):
          row_context_sequence_output[j] = tf.squeeze(
              row_context_sequence_output[j], axis=1)
          row_context_grid.append(row_context_sequence_output[j])
      if not per_row_encoding:
        header_encoding = tf.concat(header_encoding, axis=1)
        header_encoding = tf.reduce_mean(header_encoding, axis=1)
        row_context_grid = [header_encoding] + row_context_grid
    if grid_type != "row":
      if per_row_encoding:
        chunk_size = 1
        st_idx = 0
      else:
        chunk_size = 512 // max_cell_context_length - 1
        st_idx = 1

      if use_bert or use_mobilebert:
        if grid_type == "both":
          bert_scope = "col/bert"
        else:
          bert_scope = "bert"
      else:
        bert_vocab_size = bert_config.vocab_size
        with tf.variable_scope("col", reuse=tf.AUTO_REUSE):
          col_context_embedding = tf.get_variable(
              name="cell_context_embedding",
              shape=[bert_vocab_size, embedding_size],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
          col_context_encoder_cells = [model_utils.build_lstm(hidden_size)
                                       for _ in range(num_encoder_layers)]
          col_context_encoder_cells = contrib_rnn.MultiRNNCell(
              col_context_encoder_cells, state_is_tuple=True)
      for i in range(1, height, chunk_size):
        for j in range(chunk_size):
          split_col_cell_context[i + j] = tf.squeeze(
              split_col_cell_context[i + j], axis=1)
          split_col_context_mask[i + j] = tf.squeeze(
              split_col_context_mask[i + j], axis=1)
          split_col_context_segment_ids[i + j] = tf.squeeze(
              split_col_context_segment_ids[i + j], axis=1)
        if per_row_encoding:
          concat_col_cell_context = split_col_cell_context[i]
          concat_col_mask = split_col_context_mask[i]
          concat_col_segment_ids = split_col_context_segment_ids[i]
        else:
          concat_col_cell_context = tf.concat(
              [cur_col] + split_col_cell_context[i: i + chunk_size], axis=-1)
          concat_col_mask = tf.concat(
              [cur_col_mask] + split_col_context_mask[i: i + chunk_size],
              axis=-1)
          concat_col_segment_ids = tf.concat(
              [cur_col_segment_ids] +
              split_col_context_segment_ids[i: i + chunk_size],
              axis=-1)
        if use_mobilebert:
          col_bert_context_model = mobilebert_modeling.BertModel(
              config=bert_config, is_training=is_training,
              input_ids=concat_col_cell_context, input_mask=concat_col_mask,
              token_type_ids=concat_col_segment_ids,
              use_one_hot_embeddings=use_one_hot_embeddings,
              scope=bert_scope)
        elif use_bert:
          col_bert_context_model = bert_modeling.BertModel(
              config=bert_config, is_training=is_training,
              input_ids=concat_col_cell_context, input_mask=concat_col_mask,
              token_type_ids=concat_col_segment_ids,
              use_one_hot_embeddings=use_one_hot_embeddings,
              scope=bert_scope)
        else:
          cell_context_embeddings = tf.nn.embedding_lookup(
              col_context_embedding, concat_col_cell_context)
          col_context_sequence_output, _ = tf.nn.dynamic_rnn(
              col_context_encoder_cells,
              cell_context_embeddings,
              initial_state=col_context_encoder_cells.get_initial_state(
                  batch_size=batch_size, dtype=tf.float32),
              dtype=tf.float32)
        if use_bert or use_mobilebert:
          col_context_sequence_output = (
              col_bert_context_model.get_sequence_output())
        col_context_sequence_output = tf.reshape(
            col_context_sequence_output,
            [batch_size, chunk_size + st_idx, max_cell_context_length, -1])
        col_context_sequence_output = tf.split(
            col_context_sequence_output, chunk_size + st_idx, axis=1)
        for j in range(st_idx, chunk_size + st_idx):
          col_context_sequence_output[j] = tf.squeeze(
              col_context_sequence_output[j], axis=1)
          col_context_grid.append(col_context_sequence_output[j])
  if cell_context_encoding:
    with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
      if grid_type != "col":
        with tf.variable_scope("row", reuse=tf.AUTO_REUSE):
          row_context_grid = tf.stack(row_context_grid, axis=1)
          _, row_height, width, _ = bert_modeling.get_shape_list(
              row_context_grid, expected_rank=4)
          if use_cnn:
            if conv_type == "grid":
              # Adds a conv layer with hidden_size filters of size [hxw],
              # followed by the default (implicit) ReLU activation.
              conv1 = slim.conv2d(
                  row_context_grid, hidden_size,
                  [row_height, row_width],
                  padding="SAME", scope="grid_conv")
            else:
              col_conv = slim.conv2d(row_context_grid, hidden_size,
                                     [row_height, 1],
                                     padding="SAME", scope="col_conv")
              if conv_type == "cross":
                row_conv = slim.conv2d(row_context_grid, hidden_size,
                                       [1, width],
                                       padding="SAME", scope="row_conv")
                conv1 = col_conv + row_conv
              else:
                conv1 = col_conv
            # Reshapes the hidden units such that instead of 2D maps,
            # they are 1D vectors:
            row_context_sequence_output = tf.reshape(
                conv1, [batch_size, row_height * width, hidden_size])
            if skip_connection:
              row_context_grid = tf.reshape(
                  row_context_grid,
                  [batch_size, row_height * width, -1])
              row_context_sequence_output = tf.concat(
                  [row_context_sequence_output, row_context_grid], axis=-1)
          else:
            row_context_sequence_output = tf.reshape(
                row_context_grid,
                [batch_size, row_height * max_cell_context_length, -1])

          if use_pointer_network:
            batch_row_indices = tf.range(tf.to_int32(batch_size))
            batch_row_indices = tf.expand_dims(batch_row_indices, dim=-1)
            batch_row_indices = tf.repeat(
                batch_row_indices, repeats=row_height * row_width, axis=1)
            batch_row_indices = tf.reshape(
                batch_row_indices, [batch_size * row_height * row_width])
            row_indices = tf.range(tf.to_int32(row_height))
            row_indices = tf.expand_dims(row_indices, dim=-1)
            row_indices = tf.repeat(row_indices, repeats=row_width, axis=1)
            row_indices = tf.repeat(row_indices, repeats=batch_size, axis=0)
            row_indices = tf.reshape(
                row_indices, [batch_size * row_height * row_width])
            row_cell_indices = tf.reshape(
                row_cell_indices, [batch_size * row_height * row_width])
            row_cell_indices = tf.stack([batch_row_indices, row_indices,
                                         row_cell_indices], axis=1)
            row_pooled_output = tf.reshape(
                row_context_sequence_output,
                [batch_size, row_height, max_cell_context_length, -1])
            row_pooled_output = tf.gather_nd(
                row_pooled_output, row_cell_indices)
            row_pooled_output = tf.reshape(
                row_pooled_output,
                [batch_size, row_height, row_width, -1])
            if use_bert or use_mobilebert:
              pooled_linear_name = "bert_pooled_output_linear"
            else:
              pooled_linear_name = "pooled_output_linear"
            row_pooled_output = tf.layers.dense(
                row_pooled_output,
                hidden_size,
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.02),
                name=pooled_linear_name)
            row_row_pooled_output = tf.reduce_mean(
                row_pooled_output, axis=2)
            row_col_pooled_output = tf.reduce_mean(
                row_pooled_output, axis=1)
            row_row_pooled_output = tf.split(
                row_row_pooled_output, [1] * 11 + [11], axis=1)
            row_row_pooled_output = tf.concat(
                list(reversed(row_row_pooled_output[1:])), axis=1)
            row_col_pooled_output = tf.split(row_col_pooled_output,
                                             [1] * 10 + [11], axis=1)
            row_col_pooled_output = tf.concat(
                list(reversed(row_col_pooled_output)), axis=1)
            output_token_embeddings = tf.get_variable(
                name="formula_token_embedding",
                shape=[vocab_size - 42, embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_token_embeddings = tf.expand_dims(output_token_embeddings,
                                                     dim=0)
            output_token_embeddings = tf.tile(output_token_embeddings,
                                              [batch_size, 1, 1])
            output_token_embeddings = tf.split(
                output_token_embeddings,
                [constants.ROW_ID, vocab_size - 42 - constants.ROW_ID],
                axis=1)
            output_token_embeddings = tf.concat(
                [output_token_embeddings[0],
                 row_row_pooled_output, row_col_pooled_output,
                 output_token_embeddings[1]], axis=1)
          row_context_mask = tf.expand_dims(row_context_mask, axis=-1)
          row_cell_data_encoder_output = tf.layers.dense(
              row_context_sequence_output,
              hidden_size,
              activation=None,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
              name="cell_data_encoder_output")
          row_header_encoder_output = tf.layers.dense(
              row_context_sequence_output,
              hidden_size,
              activation=None,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
              name="header_encoder_output")
          if layer_norm:
            row_cell_data_encoder_output = slim.layer_norm(
                bert_modeling.gelu(row_cell_data_encoder_output))
            row_header_encoder_output = slim.layer_norm(
                bert_modeling.gelu(row_header_encoder_output))

          if not use_bert and not use_mobilebert:
            if max_pooling:
              split_row_context_segment_ids = tf.split(
                  row_context_segment_ids,
                  [max_cell_context_length, max_cell_context_length,
                   10 * max_cell_context_length, 10 * max_cell_context_length],
                  axis=1)
              split_row_context_mask = tf.split(
                  row_context_mask,
                  [max_cell_context_length, max_cell_context_length,
                   10 * max_cell_context_length, 10 * max_cell_context_length],
                  axis=1)
              row_context_segment_ids = tf.concat(
                  [split_row_context_segment_ids[0],
                   split_row_context_segment_ids[2]], axis=1)
              row_context_mask = tf.concat(
                  [split_row_context_mask[0], split_row_context_mask[2]],
                  axis=1)
            else:
              split_row_context_segment_ids = tf.split(
                  row_context_segment_ids,
                  [12 * max_cell_context_length, 10 * max_cell_context_length],
                  axis=1)
              split_row_context_mask = tf.split(
                  row_context_mask,
                  [12 * max_cell_context_length, 10 * max_cell_context_length],
                  axis=1)
              row_context_segment_ids = split_row_context_segment_ids[0]
              row_context_mask = split_row_context_mask[0]

          row_cell_data_mask = tf.cast(
              tf.expand_dims(row_context_segment_ids, dim=-1),
              dtype=tf.float32)
          row_header_mask = 1.0 - row_cell_data_mask
          row_cell_data_mask *= row_context_mask
          row_header_mask *= row_context_mask
          row_context_encoder_output = (
              row_header_encoder_output * row_header_mask
              + row_cell_data_encoder_output * row_cell_data_mask)

      if grid_type != "row":
        with tf.variable_scope("col", reuse=tf.AUTO_REUSE):
          col_context_grid = tf.stack(col_context_grid, axis=1)
          _, col_height, width, _ = bert_modeling.get_shape_list(
              col_context_grid, expected_rank=4)
          if use_cnn:
            if conv_type == "grid":
              # Adds a conv layer with hidden_size filters of size [hxw],
              # followed by the default (implicit) ReLU activation.
              conv1 = slim.conv2d(col_context_grid, hidden_size,
                                  [col_height, col_width],
                                  padding="SAME", scope="grid_conv")
            else:
              col_conv = slim.conv2d(col_context_grid, hidden_size,
                                     [col_height, 1],
                                     padding="SAME", scope="col_conv")
              if conv_type == "cross":
                row_conv = slim.conv2d(col_context_grid, hidden_size,
                                       [1, width],
                                       padding="SAME", scope="row_conv")
                conv1 = col_conv + row_conv
              else:
                conv1 = col_conv
            # Reshapes the hidden units such that instead of 2D maps,
            # they are 1D vectors:
            col_context_sequence_output = tf.reshape(
                conv1, [batch_size, col_height * width, hidden_size])
            if skip_connection:
              col_context_grid = tf.reshape(
                  col_context_grid,
                  [batch_size, col_height * width, -1])
              col_context_sequence_output = tf.concat(
                  [col_context_sequence_output, col_context_grid], axis=-1)
          else:
            col_context_sequence_output = tf.reshape(
                col_context_grid,
                [batch_size, col_height * max_cell_context_length, -1])
          if use_pointer_network:
            batch_col_indices = tf.range(tf.to_int32(batch_size))
            batch_col_indices = tf.expand_dims(batch_col_indices, dim=-1)
            batch_col_indices = tf.repeat(
                batch_col_indices, repeats=col_height * col_width, axis=1)
            batch_col_indices = tf.reshape(
                batch_col_indices, [batch_size * col_height * col_width])
            col_indices = tf.range(tf.to_int32(col_height))
            col_indices = tf.expand_dims(col_indices, dim=-1)
            col_indices = tf.repeat(col_indices, repeats=col_width, axis=1)
            col_indices = tf.repeat(col_indices, repeats=batch_size, axis=0)
            col_indices = tf.reshape(
                col_indices, [batch_size * col_height * col_width])
            col_cell_indices = tf.split(col_cell_indices,
                                        [col_width, col_height * col_width],
                                        axis=1)
            col_cell_indices = col_cell_indices[1]
            col_cell_indices = tf.reshape(
                col_cell_indices, [batch_size * col_height * col_width])
            col_cell_indices = tf.stack([batch_col_indices, col_indices,
                                         col_cell_indices], axis=1)
            col_pooled_output = tf.reshape(
                col_context_sequence_output,
                [batch_size, col_height, max_cell_context_length, -1])
            col_pooled_output = tf.gather_nd(
                col_pooled_output, col_cell_indices)
            col_pooled_output = tf.reshape(
                col_pooled_output,
                [batch_size, col_height, col_width, -1])
            if use_bert or use_mobilebert:
              pooled_linear_name = "bert_pooled_output_linear"
            else:
              pooled_linear_name = "pooled_output_linear"
            col_pooled_output = tf.layers.dense(
                col_pooled_output,
                hidden_size,
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name=pooled_linear_name)
            col_row_pooled_output = tf.reduce_mean(
                col_pooled_output, axis=1)
            col_col_pooled_output = tf.reduce_mean(
                col_pooled_output, axis=2)
            col_row_pooled_output = tf.split(
                col_row_pooled_output, [1] * 11 + [11], axis=1)
            col_row_pooled_output = tf.concat(
                list(reversed(col_row_pooled_output[1:])), axis=1)
            col_col_pooled_output = tf.split(
                col_col_pooled_output, [1] * 10 + [11], axis=1)
            col_col_pooled_output = tf.concat(
                list(reversed(col_col_pooled_output)), axis=1)
            if grid_type == "col":
              output_token_embeddings = tf.get_variable(
                  name="formula_token_embedding",
                  shape=[vocab_size - 42, embedding_size],
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
              output_token_embeddings = tf.expand_dims(
                  output_token_embeddings, dim=0)
              output_token_embeddings = tf.tile(output_token_embeddings,
                                                [batch_size, 1, 1])
              output_token_embeddings = tf.split(
                  output_token_embeddings,
                  [constants.ROW_ID, vocab_size - 42 - constants.ROW_ID],
                  axis=1)
              output_token_embeddings = tf.concat(
                  [output_token_embeddings[0],
                   col_row_pooled_output, col_col_pooled_output,
                   output_token_embeddings[1]], axis=1)
            else:
              output_token_embeddings = tf.split(
                  output_token_embeddings,
                  [constants.ROW_ID, 42, vocab_size - 42 - constants.ROW_ID],
                  axis=1)
              concat_range_embeddings = tf.concat(
                  [col_row_pooled_output, col_col_pooled_output], axis=1)
              output_token_embeddings = tf.concat(
                  [output_token_embeddings[0],
                   output_token_embeddings[1] + concat_range_embeddings,
                   output_token_embeddings[2]], axis=1)
          col_context_mask = tf.expand_dims(col_context_mask, axis=-1)
          col_cell_data_encoder_output = tf.layers.dense(
              col_context_sequence_output,
              hidden_size,
              activation=None,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
              name="cell_data_encoder_output")
          col_header_encoder_output = tf.layers.dense(
              col_context_sequence_output,
              hidden_size,
              activation=None,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
              name="header_encoder_output")

          if layer_norm:
            col_cell_data_encoder_output = slim.layer_norm(
                bert_modeling.gelu(col_cell_data_encoder_output))
            col_header_encoder_output = slim.layer_norm(
                bert_modeling.gelu(col_header_encoder_output))

          col_context_segment_ids = tf.split(
              col_context_segment_ids,
              [max_cell_context_length, 21 * max_cell_context_length],
              axis=1)
          col_context_segment_ids = col_context_segment_ids[1]
          col_context_mask = tf.split(
              col_context_mask,
              [max_cell_context_length, 21 * max_cell_context_length],
              axis=1)
          col_context_mask = col_context_mask[1]

          if not use_bert and not use_mobilebert:
            split_col_context_segment_ids = tf.split(
                col_context_segment_ids,
                [11 * max_cell_context_length, 10 * max_cell_context_length],
                axis=1)
            col_context_segment_ids = split_col_context_segment_ids[0]
            split_col_context_mask = tf.split(
                col_context_mask,
                [11 * max_cell_context_length, 10 * max_cell_context_length],
                axis=1)
            col_context_mask = split_col_context_mask[0]

          col_cell_data_mask = tf.cast(
              tf.expand_dims(col_context_segment_ids, dim=-1),
              dtype=tf.float32)
          col_header_mask = 1.0 - col_cell_data_mask
          col_cell_data_mask = col_cell_data_mask * col_context_mask
          col_header_mask = col_header_mask * col_context_mask
          col_context_encoder_output = (
              col_header_encoder_output * col_header_mask +
              col_cell_data_encoder_output * col_cell_data_mask)
      if grid_type == "row":
        context_encoder_output = row_context_encoder_output
        cell_data_mask = row_cell_data_mask
        header_mask = row_header_mask
      elif grid_type == "col":
        context_encoder_output = col_context_encoder_output
        cell_data_mask = col_cell_data_mask
        header_mask = col_header_mask
      else:
        context_encoder_output = tf.concat([row_context_encoder_output,
                                            col_context_encoder_output],
                                           axis=1)
        cell_data_mask = tf.concat([row_cell_data_mask, col_cell_data_mask],
                                   axis=1)
        header_mask = tf.concat([row_header_mask, col_header_mask], axis=1)

    context_encoder_output = tf.layers.dropout(context_encoder_output,
                                               dropout_rate,
                                               training=use_dropout,
                                               name="context_encoder_dropout")
  with tf.variable_scope("cell_index_encoder", reuse=tf.AUTO_REUSE):
    index_embedding = tf.get_variable(
        name="cell_index_embedding",
        shape=[1000, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    cell_indices = tf.concat([column_index, record_index], axis=1)
    cell_indices_embeddings = tf.nn.embedding_lookup(index_embedding,
                                                     cell_indices)

    if cell_position_encoding:
      encoder_cells = [model_utils.build_lstm(hidden_size)
                       for _ in range(num_encoder_layers)]
      encoder_cells = contrib_rnn.MultiRNNCell(encoder_cells,
                                               state_is_tuple=True)
      encoder_state = encoder_cells.get_initial_state(
          batch_size=batch_size, dtype=tf.float32)
      _, encoder_state = tf.nn.dynamic_rnn(
          encoder_cells,
          cell_indices_embeddings,
          initial_state=encoder_state,
          dtype=tf.float32)

  sketch_mask = ([0] * constants.SPECIAL_TOKEN_SIZE +
                 [1] * (vocab_size - constants.SPECIAL_TOKEN_SIZE))
  sketch_mask[constants.END_FORMULA_SKETCH_ID] = 1
  sketch_mask = tf.convert_to_tensor(np.array(sketch_mask), dtype=tf.float32)
  sketch_mask = tf.expand_dims(sketch_mask, 0)
  sketch_mask = tf.tile(sketch_mask, [batch_size, 1])

  range_mask = [0] * vocab_size
  range_mask[constants.RANGE_TOKEN_ID] = 1
  range_mask[constants.RANGE_SPLIT_ID] = 1
  range_mask[constants.END_RANGE_ID] = 1
  range_mask[constants.EOF_ID] = 1
  for i in range(constants.ROW_ID, constants.COL_ID + 21):
    range_mask[i] = 1
  range_mask = tf.convert_to_tensor(np.array(range_mask), dtype=tf.float32)
  range_mask = tf.expand_dims(range_mask, 0)
  range_mask = tf.tile(range_mask, [batch_size, 1])

  range_bool_mask = [0]
  range_bool_mask = tf.convert_to_tensor(np.array(range_bool_mask),
                                         dtype=tf.int32)
  range_bool_mask = tf.expand_dims(range_bool_mask, 0)
  range_bool_mask = tf.tile(range_bool_mask, [batch_size, 1])

  with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
    token_embedding = tf.get_variable(
        name="formula_token_embedding",
        shape=[vocab_size, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    cells = [model_utils.build_lstm(hidden_size)
             for _ in range(num_decoder_layers)]
    cells = contrib_rnn.MultiRNNCell(cells, state_is_tuple=True)

    def symbols_to_logits(partial_seqs, cur_step, beam_search_state):
      decoder_state = beam_search_state.get("decoder_state")
      input_tokens = tf.slice(
          partial_seqs, [0, cur_step,], [batch_size * beam_size, 1])

      cur_formula_mask = beam_search_state.get("formula_mask")
      cur_range_bool_mask = beam_search_state.get("range_bool_mask")

      sketch_idx = tf.constant(constants.END_FORMULA_SKETCH_ID, dtype=tf.int32)

      cur_formula_mask = tf.where(
          tf.equal(tf.reshape(input_tokens, [-1]), sketch_idx),
          tf.tile(range_mask, [beam_size, 1]), cur_formula_mask)

      cur_range_bool_mask = tf.where(
          tf.equal(tf.reshape(input_tokens, [-1]), sketch_idx),
          tf.tile(1 - range_bool_mask, [beam_size, 1]), cur_range_bool_mask)

      input_embeddings = tf.nn.embedding_lookup(
          token_embedding,
          input_tokens)

      decoder_output, decoder_state = tf.nn.dynamic_rnn(
          cells,
          input_embeddings,
          initial_state=decoder_state,
          dtype=tf.float32)

      if cell_context_encoding:
        context_encoder_output = beam_search_state.get("context_encoder_output")
        cell_data_mask = beam_search_state.get("cell_data_mask")
        header_mask = beam_search_state.get("header_mask")
        if use_pointer_network:
          output_token_embeddings = beam_search_state.get(
              "output_token_embeddings")

        if max_pooling:
          context_shape = bert_modeling.get_shape_list(
              context_encoder_output, expected_rank=3)
          height = context_shape[1] // max_cell_context_length
          context_encoder_output = tf.reshape(
              context_encoder_output,
              [batch_size * beam_size, height, max_cell_context_length, -1])
          cell_data_mask = tf.reshape(
              cell_data_mask,
              [batch_size * beam_size, height, max_cell_context_length, -1])
          header_mask = tf.reshape(
              header_mask,
              [batch_size * beam_size, height, max_cell_context_length, -1]
              )
          decoder_output = tf.expand_dims(decoder_output, axis=1)
          decoder_output = tf.repeat(
              decoder_output, repeats=height, axis=1)

        cell_data_attn_vec = tf.layers.dense(
            decoder_output,
            hidden_size,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="cell_data_encoder_attention_layer")

        cell_data_encoder_attn_w = tf.matmul(context_encoder_output,
                                             cell_data_attn_vec,
                                             transpose_b=True)
        cell_data_encoder_attn_w = tf.layers.dropout(
            cell_data_encoder_attn_w, dropout_rate, training=use_dropout,
            name="cell_data_attn_dropout")
        cell_data_encoder_attn_w -= 1e6 * (1 - cell_data_mask)
        cell_data_encoder_attn_w = tf.nn.softmax(cell_data_encoder_attn_w,
                                                 axis=-2)
        cell_data_encoder_embeddings = tf.matmul(cell_data_encoder_attn_w,
                                                 context_encoder_output,
                                                 transpose_a=True)
        cell_data_encoder_vec = tf.layers.dense(
            cell_data_encoder_embeddings,
            hidden_size,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="cell_data_encoder_linear")

        header_attn_vec = tf.layers.dense(
            decoder_output,
            hidden_size,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="header_encoder_attention_layer")
        header_encoder_attn_w = tf.matmul(context_encoder_output,
                                          header_attn_vec,
                                          transpose_b=True)
        header_encoder_attn_w = tf.layers.dropout(
            header_encoder_attn_w, dropout_rate, training=use_dropout,
            name="header_attn_dropout")
        header_encoder_attn_w -= 1e6 * (1 - header_mask)
        header_encoder_attn_w = tf.nn.softmax(header_encoder_attn_w,
                                              axis=-2)
        header_encoder_embeddings = tf.matmul(header_encoder_attn_w,
                                              context_encoder_output,
                                              transpose_a=True)
        header_encoder_vec = tf.layers.dense(
            header_encoder_embeddings,
            hidden_size,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="header_encoder_linear")

        decoder_vec = tf.layers.dense(
            decoder_output,
            hidden_size,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="decoder_linear")

        decoder_output = tf.concat(
            [cell_data_encoder_vec, header_encoder_vec, decoder_vec], axis=-1)

      sketch_logits = tf.layers.dense(
          decoder_output,
          vocab_size,
          activation=None,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
          name="formula_sketch_logit")

      if use_pointer_network:
        range_attn_vec = tf.layers.dense(
            decoder_output,
            hidden_size,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="range_attention_layer")
        range_logits = tf.matmul(output_token_embeddings,
                                 range_attn_vec,
                                 transpose_b=True)
        range_logits = tf.squeeze(range_logits, axis=-1)
      else:
        range_logits = tf.layers.dense(
            decoder_output,
            vocab_size,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="range_logit")
        range_logits = tf.squeeze(range_logits, axis=-2)

      if two_stage_decoding:
        pred_logits = tf.where(
            tf.equal(tf.reshape(cur_range_bool_mask, [-1]), 0),
            tf.squeeze(sketch_logits, axis=-2),
            range_logits)
      elif use_pointer_network:
        pred_logits = range_logits
      else:
        pred_logits = tf.squeeze(sketch_logits, axis=-2)

      if cell_context_encoding and max_pooling:
        pred_logits = tf.reduce_max(pred_logits, axis=1)

      if two_stage_decoding:
        pred_logits -= 1e6 * (1 - cur_formula_mask)

      beam_search_state.update({
          "decoder_state": decoder_state,
          "formula_mask": cur_formula_mask,
          "range_bool_mask": cur_range_bool_mask
      })
      return pred_logits, beam_search_state

    logits = []
    if cell_context_encoding:
      decoder_state = cells.get_initial_state(
          batch_size=batch_size, dtype=tf.float32)
      _, decoder_state = tf.nn.dynamic_rnn(
          cells,
          cell_indices_embeddings,
          initial_state=decoder_state,
          dtype=tf.float32)
    elif cell_position_encoding:
      decoder_state = encoder_state
    else:
      decoder_state = cells.get_initial_state(
          batch_size=batch_size, dtype=tf.float32)
    beam_search_state = {
        "decoder_state": decoder_state,
        "formula_mask": sketch_mask,
        "range_bool_mask": range_bool_mask,
    }
    if cell_context_encoding:
      beam_search_state.update({
          "context_encoder_output": context_encoder_output,
          "cell_data_mask": cell_data_mask,
          "header_mask": header_mask
      })
    if use_pointer_network:
      beam_search_state.update({
          "output_token_embeddings": output_token_embeddings
      })
    initial_input_tokens = tf.constant(
        constants.GO_ID, dtype=tf.int32, shape=(1,), name="GO_ids")
    initial_input_tokens = tf.tile(initial_input_tokens, [batch_size])

    if is_training:
      initial_input_tokens = tf.expand_dims(initial_input_tokens, axis=1)
      full_formula = tf.concat([initial_input_tokens, formula], axis=1)
      for cur_step in range(formula_length):
        partial_seqs = tf.slice(
            full_formula, [0, 0], [batch_size, cur_step + 1])
        pred_logits, beam_search_state = symbols_to_logits(
            partial_seqs, cur_step, beam_search_state)
        logits.append(pred_logits)
      logits = tf.stack(logits, axis=1)
      return logits
    elif beam_size <= 1:
      input_tokens = initial_input_tokens
      for cur_step in range(formula_length):
        pred_logits, beam_search_state = symbols_to_logits(
            input_tokens, cur_step, beam_search_state)
        pred_logits = tf.squeeze(pred_logits, axis=1)
        logits.append(pred_logits)
        input_tokens = tf.argmax(pred_logits, axis=-1,
                                 output_type=tf.int32)
      logits = tf.stack(logits, axis=1)
      return logits
    else:
      initial_input_tokens = tf.expand_dims(initial_input_tokens, axis=1)
      full_formula = tf.concat([initial_input_tokens, formula], axis=1)
      for cur_step in range(formula_prefix_length):
        cur_tokens = tf.slice(
            full_formula, [0, cur_step], [batch_size, 1])
        cur_embeddings = tf.nn.embedding_lookup(
            token_embedding,
            cur_tokens)
        _, decoder_state = tf.nn.dynamic_rnn(
            cells,
            cur_embeddings,
            initial_state=decoder_state,
            dtype=tf.float32)
      beam_search_state.update({"decoder_state": decoder_state})
      initial_input_tokens = tf.slice(
          full_formula, [0, formula_prefix_length], [batch_size, 1])
      initial_input_tokens = tf.squeeze(initial_input_tokens, axis=1)
      beam_seqs, beam_probs, _ = beam_search.beam_search(
          symbols_to_logits_fn=symbols_to_logits,
          initial_ids=initial_input_tokens,
          beam_size=beam_size,
          decode_length=formula_length - formula_prefix_length,
          vocab_size=vocab_size,
          alpha=1.0,
          states=beam_search_state,
          eos_id=constants.EOF_ID,
          stop_early=False,
          use_tpu=use_tpu)
      if formula_prefix_length > 0:
        initial_partial_seqs = tf.slice(
            full_formula, [0, 0], [batch_size, formula_prefix_length])
        initial_partial_seqs = tf.expand_dims(initial_partial_seqs, axis=1)
        initial_partial_seqs = tf.tile(initial_partial_seqs, [1, beam_size, 1])
        beam_seqs = tf.concat([initial_partial_seqs, beam_seqs], axis=2)
      return beam_seqs, beam_probs
