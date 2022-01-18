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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for working with tf.Dataset datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config
import mutation_encoding
import numpy
import pandas
import residue_encoding
import tensorflow as tf


ONEHOT_VARLEN_SEQUENCE_ENCODER = mutation_encoding.DirectSequenceEncoder(
    residue_encoding.ResidueIdentityEncoder(config.RESIDUES))

ONEHOT_FIXEDLEN_MUTATION_ENCODER = mutation_encoding.MutationSequenceEncoder(
    residue_encoding.ResidueIdentityEncoder(config.RESIDUES),
    config.R1_TILE21_WT_SEQ)


def as_tf_example(example):
  """Converts a dict-based example to a tf.Example proto.

  Args:
    example: (dict) A dict of attributes describing a single example.
  Returns:
    (tf.Example) A proto representation of the example.
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'sequence': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[example['sequence']])),
      'mutation_sequence': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[example['mutation_sequence']])),
      'partition': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[example['partition']])),
      'is_viable': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[int(example['is_viable'])])),
      'num_mutations': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[example['num_mutations']])),
      'viral_selection': tf.train.Feature(
          float_list=tf.train.FloatList(value=[example['viral_selection']])),
  }))


def parse_tf_example(tf_example_str):
  """Parses a tf.Example proto string.

  Args:
    tf_example_str: (str) A serialized tf.Example proto.
  Returns:
    ({str: Tensor}) A dict of Tensors representing each field in the example.
  """
  return tf.parse_single_example(
      serialized=tf_example_str,
      features={
          'sequence': tf.FixedLenFeature([], dtype=tf.string),
          'mutation_sequence': tf.FixedLenFeature([], dtype=tf.string),
          'partition': tf.FixedLenFeature([], dtype=tf.string),
          'is_viable': tf.FixedLenFeature([], dtype=tf.int64),
          'num_mutations': tf.FixedLenFeature([], dtype=tf.int64),
          'viral_selection': tf.FixedLenFeature([], dtype=tf.float32),
      },
  )


def generate_examples(df):
  """Maps dataframe columns to example attributes.

  Args:
    df: (pandas.DataFrame) A dataframe.
  Yields:
    (tf.Example>) A stream of tf.Example protos.
  """
  for row in df.itertuples():
    yield as_tf_example({
        'sequence': row.aa_seq,
        'mutation_sequence': row.mask,
        'partition': row.partition,
        'is_viable': row.is_viable,
        'num_mutations': row.mut,
        'viral_selection': row.S_clipped,
    })


def as_dataframe(dataset, batch_size=1024):
  """Converts a tf.Dataset of examples to a corresponding dataframe.

  Args:
    dataset: (tf.Dataset) A "row-like"/flat dataset where each example is a
      {key: Tensor} dict and each Tensor value is scalar.
    batch_size: (int) Tuning parameter for the batch size that will be used to
      internally enumerate the examples within a tf.Session.
  Returns:
    (pandas.DataFrame) A dataframe containing the dataset.
  """
  # Note pulling examples in batches is done here purely for efficiency, versus
  # pulling examples one-by-one.
  it = dataset.batch(batch_size).make_one_shot_iterator()
  examples = None
  with tf.Session() as sess:
    while True:
      try:
        batch_examples = sess.run(it.get_next())
        if examples is None:
          examples = batch_examples
        else:
          for key, series in examples.iteritems():
            examples[key] = numpy.concatenate([series, batch_examples[key]])  # pylint: disable=unsupported-assignment-operation
      except tf.errors.OutOfRangeError:
        break

  return pandas.DataFrame(examples)


def write_tfrecord_dataset(filepath, examples):
  """Writes the given example protos to TFRecord format.

  Args:
    filepath: (str) The path to the file to create.
    examples: (iter<tf.Example>) An iterable of tf.Example protos.
  """
  with tf.python_io.TFRecordWriter(filepath) as writer:
    for example in examples:
      writer.write(example.SerializeToString())


def read_tfrecord_dataset(filepaths):
  """Reads a dataset of tf.Example protos from TFRecord-formatted files.

  Args:
    filepaths: (list<str>) A list of file paths to read.
  Returns:
    (tf.Dataset) A tf.Dataset of examples; see `parse_tf_example` for the
    structure of each example returned.
  """
  return tf.data.TFRecordDataset(filenames=filepaths).map(parse_tf_example)


def encode_fixedlen(example, encoder, sequence_key='mutation_sequence'):
  """Encodes an example to a fixed length vector respresentation.

  The fixed length representation is useful for models that need all examples to
  have the same shape.

  Note: fixed-length encoding used for all CNN and LR models.

  Args:
    example: ({str: Tensor}) A flat dict of Tensor-valued attributes; e.g., as
      returned by `tf.parse_single_example`.
    encoder: (SequenceEncoder) A sequence encoder that encodes a sequence string
      to a fixed length feature vector representation.
    sequence_key: (str) The name of the example attribute that should be used
      for encoding; e.g., 'sequence' or 'mutation_sequence'.
  Returns:
    (features, label) A tuple of tensors corresponding to the features and label
    for the example.
  """
  encoded_sequence = tf.cast(
      tf.py_func(encoder.encode, [example[sequence_key]], tf.float64),
      tf.float32)

  features = {
      'sequence': encoded_sequence,
  }

  label = tf.cast(example['is_viable'], tf.int64)

  return features, label


def encode_varlen(example, encoder, sequence_key='sequence'):
  """Encodes an example to a variable length vector representation.

  Note: variable-length encoding used for all RNN models.

  Args:
    example: ({str: Tensor}) A flat dict of Tensor-valued attributes; e.g., as
      returned by `tf.parse_single_example`.
    encoder: (SequenceEncoder) A sequence encoder that encodes a sequence string
      to a variable length feature vector representation.
    sequence_key: (str) The name of the example attribute that should be used
      for encoding; e.g., 'sequence' or 'mutation_sequence'.
  Returns:
    (features, label) A tuple of tensors corresponding to the features and label
    for the example.
  """
  encoded_sequence = tf.cast(
      tf.py_func(encoder.encode, [example[sequence_key]], tf.float64),
      tf.float32)

  features = {
      'sequence': encoded_sequence,
      'sequence_length': tf.shape(encoded_sequence)[0],
  }

  label = tf.cast(example['is_viable'], tf.int64)

  return features, label


def as_estimator_input_fn(
    dataset,
    batch_size,
    num_epochs=None,
    shuffle=False,
    drop_partial_batches=False,
    sequence_element_encoding_shape=None):
  """Creates a tf.Estimator input_fn from the given tf.dataset.

  Args:
    dataset: (tf.dataset) A dataset.
    batch_size: (int) The batch size.
    num_epochs: (int or None) The number of epochs to iterate the dataset, or
      None to indicate that the dataset should repeat indefinitely.
    shuffle: (bool) Should the dataset be shuffled?
    drop_partial_batches: (bool) Should batches of size less than `batch_size`
      be dropped?
    sequence_element_encoding_shape: (int) The shape of a single element of the
      sequence.
  Returns:
    (fn) A tf.Estimator model input_fn.
  """

  def input_fn():
    """A tf.Estimator input_fn."""
    ds = dataset.shuffle(buffer_size=10000) if shuffle else dataset

    if num_epochs is None:
      ds = ds.repeat()
    else:
      ds = ds.repeat(num_epochs)

    if sequence_element_encoding_shape is None:
      ds = ds.batch(batch_size)
    else:
      padding_spec = (
          # Features padding spec (first element of 2-tuple).
          {
              'sequence': [
                  None,  # Each sequence varies in length (None==unknown).
                  sequence_element_encoding_shape,  # Elements have fixed size.
              ],
              'sequence_length': []  # One scalar value per example.
          },
          # Label padding spec (second element of 2-tuple).
          [],  # One scalar label value per example.
      )
      ds = ds.padded_batch(batch_size, padding_spec)

    if drop_partial_batches:
      ds = ds.filter(
          lambda batch_features, batch_label:  # pylint: disable=g-long-lambda
          tf.equal(tf.shape(batch_features['sequence'])[0], batch_size))

    iterator = ds.make_one_shot_iterator()
    features_tensor, labels_tensor = iterator.get_next()
    return features_tensor, labels_tensor

  return input_fn

