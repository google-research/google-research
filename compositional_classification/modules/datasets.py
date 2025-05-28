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

# coding=utf-8
"""Utility functions to load the different dataset versions."""

import os

import tensorflow as tf

# Special tokens
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
OOV_TOKEN_ID = 2
SEP_TOKEN_ID = 3
CLS_TOKEN_ID = 4

# Segment embedding vocab (0: quesion, 1: query)
SEGMENT_VOCAB_SIZE = 2

# Relative attention
RELATIVE_VOCAB_SIZE = 10
RELATIVE_ATT_ID_NORMAL = 0  # 0: no special relation
RELATIVE_ATT_ID_SELF = 1  # 1: self attention
RELATIVE_ATT_ID_FROM_GLOBAL_CLS = 2  # 2, 3: attention from/to global CLS token
RELATIVE_ATT_ID_TO_GLOBAL_CLS = 3
RELATIVE_ATT_ID_PARENT_TO_CHILD = 4  # 4: parent to child
RELATIVE_ATT_ID_CHILD_TO_PARENT = 5  # 5: child to parent
RELATIVE_ATT_ID_BLOCK_QUESTION = 6  # 6: block attention among question tokens
RELATIVE_ATT_ID_BLOCK_QUERY = 7  # 7: block attention among query tokens
RELATIVE_ATT_ID_XLINK_QS_TO_QR = 8  # 8: same-entity cross link (qs -> qr)
RELATIVE_ATT_ID_XLINK_QR_TO_QS = 9  # 9: same-entity cross link (qr -> qs)


def load_cls_dataset(hparams,
                     data_root,
                     name='train',
                     shuffle=None):
  """Reads tfrecord file and converts to binary classification dataset."""
  # Reads tfrecord file
  data_fpath = os.path.join(data_root, name + '.tfrecord')
  raw_dataset = tf.data.TFRecordDataset(filenames=[data_fpath])

  # Parses examples to produce (sequence, label) dataset
  feature_description = {
      'question':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'query':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'label':
          tf.io.FixedLenFeature([], tf.int64),
  }

  def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    sentence = [example['question'], [SEP_TOKEN_ID], example['query']]
    if hparams.add_cls_token:
      sentence.insert(0, [CLS_TOKEN_ID])
    if hparams.add_eos_token:
      sentence.append([EOS_TOKEN_ID])
    sentence = tf.concat(sentence, -1)

    example = (sentence, example['label'])
    return example

  # Shuffles and repeats dataset if train mode
  if shuffle or (name == 'train' and shuffle is None):
    dataset = raw_dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)
    dataset = dataset.shuffle(buffer_size=hparams.batch_size * 10).repeat()
  else:
    dataset = raw_dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True)

  # Makes a (padded) batch
  dataset = dataset.padded_batch(
      hparams.batch_size,
      padded_shapes=([None], []),
      padding_values=(PAD_TOKEN_ID, None))
  return dataset


def load_cls_nomask_dataset(hparams,
                            data_root,
                            name='train',
                            shuffle=None):
  """Load non-tree dataset for Relative Transformer."""
  # Reads tfrecord file
  data_fpath = os.path.join(data_root, name + '.tfrecord')
  raw_dataset = tf.data.TFRecordDataset(filenames=[data_fpath])

  # Parses examples to produce (sequence, label) dataset
  feature_description = {
      'question':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'query':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'label':
          tf.io.FixedLenFeature([], tf.int64),
  }

  def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    question = example['question']
    query = example['query']
    qs_len = tf.shape(question)[0]
    qr_len = tf.shape(query)[0]

    # Input sentence
    sentence = [[CLS_TOKEN_ID], example['question'], [SEP_TOKEN_ID],
                example['query']]
    sentence_len = qs_len + qr_len + 2
    if hparams.add_eos_token:
      sentence.append([EOS_TOKEN_ID])
      sentence_len += 1
    sentence = tf.concat(sentence, -1)

    # Positional ids: only original
    pos_ids = tf.range(sentence_len)  # Pos indices increase to the end
    question_mask = tf.ones(sentence_len, tf.int32)
    additional_mask = tf.zeros(sentence_len, tf.int32)
    query_mask = tf.zeros(sentence_len, tf.int32)

    # Segment ids
    seg_ids = [tf.zeros(1 + qs_len, tf.int32), tf.ones(1 + qr_len, tf.int32)]
    if hparams.add_eos_token:
      seg_ids.append([1])  # The EOS token belongs to the second segment
    seg_ids = tf.concat(seg_ids, -1)

    # Attention mask
    mask_shape = [tf.shape(sentence)[0], tf.shape(sentence)[0]]
    att_mask = tf.ones(mask_shape, tf.int32)  # One means no mask
    relative_att_ids = tf.zeros(mask_shape, tf.int32)

    example = ({
        'token_ids': sentence,
        'attention_mask': att_mask,
        'relative_att_ids': relative_att_ids,
        'position_ids': pos_ids,
        'question_mask': question_mask,
        'query_mask': query_mask,
        'additional_mask': additional_mask,
        'segment_ids': seg_ids
    }, example['label'])
    return example

  # Shuffles and repeats dataset if train mode
  if shuffle or (name == 'train' and shuffle is None):
    dataset = raw_dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)
    dataset = dataset.shuffle(buffer_size=hparams.batch_size * 10).repeat()
  else:
    dataset = raw_dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True)

  # Makes a (padded) batch
  dataset = dataset.padded_batch(
      hparams.batch_size,
      padded_shapes=({
          'token_ids': [None],
          'attention_mask': [None, None],
          'relative_att_ids': [None, None],
          'position_ids': [None],
          'question_mask': [None],
          'query_mask': [None],
          'additional_mask': [None],
          'segment_ids': [None]
      }, []),
      padding_values=({
          'token_ids': PAD_TOKEN_ID,
          'attention_mask': 0,
          'relative_att_ids': 0,
          'position_ids': PAD_TOKEN_ID,
          'question_mask': 0,
          'query_mask': 0,
          'additional_mask': 0,
          'segment_ids': PAD_TOKEN_ID
      }, None))
  return dataset


def load_cls_mask_dataset(hparams,
                          data_root,
                          name='train',
                          shuffle=None):
  """Reads tfrecord file and converts to classification dataset with tree mask."""
  if not hparams.add_cls_token:
    raise ValueError('CLS token is mandatory')
  if hparams.parse_tree_attention and hparams.block_attention:
    raise ValueError('Only one of parse tree and block attention can be set')

  # Reads tfrecord file
  data_fpath = os.path.join(data_root, name + '.tfrecord')
  raw_dataset = tf.data.TFRecordDataset(filenames=[data_fpath])

  # Parses examples to produce (sequence, label) dataset
  feature_description = {
      'question':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'question_structure_tokens':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'question_tree':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'question_group':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'query':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'query_tree':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'query_group':
          tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
      'label':
          tf.io.FixedLenFeature([], tf.int64),
  }

  def _parse_function(example_proto):
    # This nested function parses an example and generates input sequences
    # (sentence, positional ids, segment ids) and attentions (attention mask,
    # relative attention ids).
    example = tf.io.parse_single_example(example_proto, feature_description)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    question = example['question']
    question_struct = example['question_structure_tokens']
    question_tree = example['question_tree']
    query = example['query']
    query_tree = example['query_tree']

    qs_len = tf.shape(question)[0]
    qs_struct_len = tf.shape(question_struct)[0]
    qr_len = tf.shape(query)[0]
    sentence_len = qs_len + qs_struct_len + qr_len + 2

    # Sentence, positional ids, and segment ids
    sentence = [[CLS_TOKEN_ID], question, question_struct, [SEP_TOKEN_ID],
                query]

    # Positional ids
    if not hparams.restart_query_pos:
      pos_ids = [tf.range(sentence_len)]  # Pos indices increase to the end
    else:
      if hparams.unique_structure_token_pos:
        # The same index (0) for structure tokens
        pos_ids = [tf.range(qs_len + 1), tf.zeros(qs_struct_len, tf.int32)]
      else:
        # Pos indices for structure tokens continues the previous indices
        pos_ids = [tf.range(qs_len + qs_struct_len + 1)]
      pos_ids.append(tf.range(qr_len + 1))  # Separate indices for query

    # Masks (question / question structure / query) for positional indices
    if not hparams.restart_query_pos:
      question_mask = [tf.ones(sentence_len, tf.int32)]
      additional_mask = [tf.zeros(sentence_len, tf.int32)]
      query_mask = [tf.zeros(sentence_len, tf.int32)]
    else:
      question_mask = [tf.ones(qs_len + 1, tf.int32)]
      additional_mask = [tf.zeros(qs_len + 1, tf.int32)]
      if hparams.unique_structure_token_pos:
        question_mask.append(tf.zeros(qs_struct_len, tf.int32))
        additional_mask.append(tf.ones(qs_struct_len, tf.int32))
      else:
        question_mask.append(tf.ones(qs_struct_len, tf.int32))
        additional_mask.append(tf.zeros(qs_struct_len, tf.int32))
      question_mask.append(tf.zeros(qr_len + 1, tf.int32))
      additional_mask.append(tf.zeros(qr_len + 1, tf.int32))
      query_mask = [
          tf.zeros(qs_len + qs_struct_len + 1, tf.int32),
          tf.ones(qr_len + 1, tf.int32)
      ]

    # Segment ids
    seg_ids = [
        tf.zeros(1 + qs_len + qs_struct_len, tf.int32),
        tf.ones(1 + qr_len, tf.int32)
    ]

    # EOS option
    if hparams.add_eos_token:
      sentence.append([EOS_TOKEN_ID])
      seg_ids.append([1])  # The EOS token belongs to the second segment
      if hparams.restart_query_pos:
        # Pos id is drawn from the query pos embedding
        pos_ids.append([qr_len + 1])
        question_mask.append([0])
        additional_mask.append([0])
        query_mask.append([1])
      else:
        # Pos id drawn from the question pos embedding (the only pos embedding)
        pos_ids.append([sentence_len])
        question_mask.append([1])
        additional_mask.append([0])
        query_mask.append([0])

    sentence = tf.concat(sentence, -1)
    pos_ids = tf.concat(pos_ids, -1)
    question_mask = tf.concat(question_mask, -1)
    additional_mask = tf.concat(additional_mask, -1)
    query_mask = tf.concat(query_mask, -1)
    seg_ids = tf.concat(seg_ids, -1)

    # Parse tree attention
    if hparams.parse_tree_attention:
      # Concatenate edges of the two parse trees
      root_edges = [[0, 1 + tf.shape(question)[0]],
                    [
                        0,
                        2 + tf.shape(question)[0] + tf.shape(question_struct)[0]
                    ]]
      question_offset = 1
      question_edges = tf.reshape(question_tree, [-1, 2]) + question_offset
      query_offset = 2 + tf.shape(question)[0] + tf.shape(question_struct)[0]
      query_edges = tf.reshape(query_tree, [-1, 2]) + query_offset

      child_edges = tf.concat([root_edges, question_edges, query_edges], 0)
      parent_edges = tf.reverse(child_edges, [1])  # (i, j) -> (j, i)
      edges = tf.concat([child_edges, parent_edges], 0)  # bidirectional edge
    mask_shape = [tf.shape(sentence)[0], tf.shape(sentence)[0]]

    # Block attention
    if hparams.block_attention:
      question_block = [[0],
                        tf.ones(qs_len + qs_struct_len, tf.int32), [0],
                        tf.zeros(qr_len, tf.int32)]
      query_block = [[0],
                     tf.zeros(qs_len + qs_struct_len, tf.int32),
                     [int(hparams.block_attention_sep)],
                     tf.ones(qr_len, tf.int32)]
      if hparams.add_eos_token:
        question_block.append([0])
        query_block.append([0])
      question_block = tf.concat(question_block, -1)
      query_block = tf.concat(query_block, -1)
      block_att_question = (tf.expand_dims(question_block, axis=0) *
                            tf.expand_dims(question_block, axis=1))
      block_att_query = (tf.expand_dims(query_block, axis=0) *
                         tf.expand_dims(query_block, axis=1))

    # Entity cross link
    if hparams.entity_cross_link:
      if not hparams.cross_link_exact:
        question_group = example['question_group']
        query_group = example['query_group']
      else:
        question_group = question
        query_group = query

      # Make cross links of the same group using broadcast
      group_ids_1 = [[0], question_group,
                     tf.zeros(qs_struct_len + qr_len + 1, tf.int32)]
      group_ids_2 = [
          tf.zeros(qs_len + qs_struct_len + 2, tf.int32), query_group
      ]
      if hparams.add_eos_token:
        group_ids_1.append([0])
        group_ids_2.append([0])
      group_ids_1 = tf.expand_dims(tf.concat(group_ids_1, -1), axis=1)
      group_ids_2 = tf.expand_dims(tf.concat(group_ids_2, -1), axis=0)
      xlink_qs_to_qr = (tf.cast(group_ids_1 == group_ids_2, tf.int32) *
                        tf.minimum(group_ids_1 * group_ids_2, 1))
      xlink_qr_to_qs = tf.transpose(xlink_qs_to_qr)

    # Attention from/to global CLS token
    if hparams.cls_global_token:  # Add one to the first row and the first col
      one_end = tf.concat(
          [[1], tf.zeros(tf.shape(sentence)[0] - 1, tf.int32)], axis=-1)
      first_row_one = tf.expand_dims(one_end, axis=0)
      first_col_one = tf.expand_dims(one_end, axis=1)

    # Attention mask
    if hparams.use_attention_mask:
      att_mask = tf.eye(tf.shape(sentence)[0], dtype=tf.int32)  # Self mask
      if hparams.cls_global_token:  # Global CLS
        att_mask += first_row_one + first_col_one
      if hparams.parse_tree_attention:  # Parse trees
        att_mask += tf.scatter_nd(edges, tf.ones(tf.shape(edges)[0], tf.int32),
                                  mask_shape)
      if hparams.block_attention:
        att_mask += block_att_question + block_att_query
      if hparams.entity_cross_link:  # Entity cross link
        att_mask += xlink_qs_to_qr + xlink_qr_to_qs
      att_mask = tf.minimum(att_mask, 1)  # Clip to one
    else:
      att_mask = tf.ones(mask_shape, tf.int32)  # One means no mask

    # Relative attention ids
    relative_att_ids = tf.zeros(mask_shape, tf.int32)
    if hparams.use_relative_attention:
      # Attention from/to the Global CLS ((0, 0) element is overwritten below)
      if hparams.cls_global_token:
        relative_att_ids += first_row_one * RELATIVE_ATT_ID_FROM_GLOBAL_CLS
        relative_att_ids += first_col_one * RELATIVE_ATT_ID_TO_GLOBAL_CLS
      # Block attention
      if hparams.block_attention:
        relative_att_ids += block_att_question * RELATIVE_ATT_ID_BLOCK_QUESTION
        relative_att_ids += block_att_query * RELATIVE_ATT_ID_BLOCK_QUERY
      # Self relative attention
      relative_att_ids = tf.tensor_scatter_nd_update(
          relative_att_ids,
          tf.tile(tf.expand_dims(tf.range(sentence_len), axis=1), [1, 2]),
          tf.ones(sentence_len, tf.int32) * RELATIVE_ATT_ID_SELF)
      # Parse tree
      if hparams.parse_tree_attention:
        relative_att_ids = tf.tensor_scatter_nd_update(
            relative_att_ids, child_edges,
            RELATIVE_ATT_ID_PARENT_TO_CHILD *
            tf.ones(tf.shape(child_edges)[0], tf.int32))
        relative_att_ids = tf.tensor_scatter_nd_update(
            relative_att_ids, parent_edges,
            RELATIVE_ATT_ID_CHILD_TO_PARENT *
            tf.ones(tf.shape(parent_edges)[0], tf.int32))
      # Cross link by addition (not overlapped to other rel att)
      if hparams.entity_cross_link:
        relative_att_ids += RELATIVE_ATT_ID_XLINK_QS_TO_QR * xlink_qs_to_qr
        relative_att_ids += RELATIVE_ATT_ID_XLINK_QR_TO_QS * xlink_qr_to_qs

    example = ({
        'token_ids': sentence,
        'attention_mask': att_mask,
        'relative_att_ids': relative_att_ids,
        'position_ids': pos_ids,
        'question_mask': question_mask,
        'query_mask': query_mask,
        'additional_mask': additional_mask,
        'segment_ids': seg_ids
    }, example['label'])
    return example

  # Shuffles and repeats dataset if train mode
  if shuffle or (name == 'train' and shuffle is None):
    dataset = raw_dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)
    dataset = dataset.shuffle(buffer_size=hparams.batch_size * 10).repeat()
  else:
    dataset = raw_dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True)

  # Makes a (padded) batch
  dataset = dataset.padded_batch(
      hparams.batch_size,
      padded_shapes=({
          'token_ids': [None],
          'attention_mask': [None, None],
          'relative_att_ids': [None, None],
          'position_ids': [None],
          'question_mask': [None],
          'query_mask': [None],
          'additional_mask': [None],
          'segment_ids': [None]
      }, []),
      padding_values=({
          'token_ids': PAD_TOKEN_ID,
          'attention_mask': 0,
          'relative_att_ids': 0,
          'position_ids': PAD_TOKEN_ID,
          'question_mask': 0,
          'query_mask': 0,
          'additional_mask': 0,
          'segment_ids': PAD_TOKEN_ID
      }, None))
  return dataset
