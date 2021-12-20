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

"""Tests for the DatasetBuilder."""

from typing import Sequence
from unittest import mock

import gin
import tensorflow as tf

from dedal import multi_task
from dedal import vocabulary
from dedal.data import builder
from dedal.data import loaders


GIN_CONFIG = """
import dedal.data.specs
import dedal.data.nlp

SEQUENCE_LENGTH = 1024
VOCAB = %vocabulary.alternative
SEQUENCE_KEY = 'sequence'
DatasetBuilder.data_loader = @TFDSLoader()
TFDSLoader.name = 'fake'
TFRecordsLoader.output_sequence_key = %SEQUENCE_KEY

DatasetBuilder.transformations = [
    @EOS(),
    @CropOrPad(),
    @DynamicLanguageModelMasker(),
]
CropOrPad.size = %SEQUENCE_LENGTH
CropOrPad.random = True
DynamicLanguageModelMasker.on = %SEQUENCE_KEY
DynamicLanguageModelMasker.out = [%SEQUENCE_KEY, 'target', 'weights']
DatasetBuilder.labels = @labels/multi_task.Backbone()
labels/multi_task.Backbone.embeddings = [('target', 'weights')]
DatasetBuilder.metadata = ('seq_key',)
"""


def make_fake_sequence_dataset(num_examples = 1000):
  voc = vocabulary.alternative
  sampler = vocabulary.Sampler(voc)
  ds = tf.data.Dataset.from_tensor_slices({
      'sequence': sampler.sample((num_examples, 128)),
      'seq_key': tf.range(num_examples, dtype=tf.int32),
      'fam_key': tf.range(num_examples, 2 * num_examples, dtype=tf.int32),
  })
  return ds


class DatasetBuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()
    gin.parse_config(GIN_CONFIG)
    self.addCleanup(mock.patch.stopall)
    self.mock_load = mock.patch.object(
        loaders.TFDSLoader, 'load', autospec=True).start()

  def test_transforms_from_gin(self):
    self.mock_load.return_value = make_fake_sequence_dataset()
    ds_builder = builder.DatasetBuilder()
    batch = 32
    ds = ds_builder.make('test', batch)
    inputs, y_true, weights, metadata = next(iter(ds))
    self.assertIsInstance(inputs, tf.Tensor)
    self.assertEqual(inputs.dtype, tf.int32)
    self.assertIsInstance(y_true, dict)
    self.assertGreater(len(y_true), 0)
    self.assertIsInstance(weights, dict)
    self.assertGreater(len(weights), 0)
    for y in multi_task.Backbone.unflatten(y_true):
      self.assertEqual(y.dtype, tf.int32)
      self.assertEqual(y.shape, (batch, 1024))  # from gin.
    for y in multi_task.Backbone.unflatten(weights):
      self.assertEqual(y.dtype, tf.float32)
    self.assertIn('seq_key', metadata)
    self.assertIsInstance(metadata['seq_key'], tf.Tensor)
    self.assertEqual(metadata['seq_key'].dtype, tf.int32)
    self.assertNotIn('fam_key', metadata)


GIN_CONFIG_MULTI = """
import dedal.data.loaders
import dedal.data.nlp

SEQUENCE_LENGTH1 = 1024
SEQUENCE_LENGTH2 = 512

vocabulary.get_default.vocab = %vocabulary.alternative
TFRecordsLoader.output_sequence_key = 'sequence'

# Configures multi-input topology.
multi_task.SwitchBackbone.embeddings = [1, 0]
multi_task.SwitchBackbone.alignments = []

MultiDatasetBuilder.builders = [
  @uniref/builder.DatasetBuilder(),
  @pfam/builder.DatasetBuilder(),
]
MultiDatasetBuilder.switch = @multi_task.SwitchBackbone()

# Configures first `DatasetBuilder`.
uniref/DatasetBuilder.data_loader = @TFDSLoader()
TFDSLoader.name = 'fake'
uniref/DatasetBuilder.transformations = [
    @uniref/EOS(),
    @uniref/CropOrPad(),
    @uniref/DynamicLanguageModelMasker(),
]
uniref/CropOrPad.size = %SEQUENCE_LENGTH1
uniref/CropOrPad.random = True
uniref/DynamicLanguageModelMasker.on = 'sequence'
uniref/DynamicLanguageModelMasker.out = ['sequence', 'target', 'weights']
uniref/DatasetBuilder.labels = @uniref/labels/multi_task.Backbone()
uniref/labels/multi_task.Backbone.embeddings = [('target', 'weights')]
uniref/DatasetBuilder.metadata = ('seq_key',)

# Configures second `DatasetBuilder`.
pfam/DatasetBuilder.data_loader = @CSVLoader()
CSVLoader.folder = 'fake_folder'
CSVLoader.fields = {}
CSVLoader.fields_to_use = []
pfam/DatasetBuilder.transformations = [
    @pfam/Encode(),
    @pfam/EOS(),
    @pfam/CropOrPad(),
    @pfam/DynamicLanguageModelMasker(),
]
pfam/CropOrPad.size = %SEQUENCE_LENGTH2
pfam/CropOrPad.random = False
pfam/DynamicLanguageModelMasker.on = 'sequence'
pfam/DynamicLanguageModelMasker.out = ['sequence', 'target', 'weights']
pfam/DatasetBuilder.labels = @uniref/labels/multi_task.Backbone()
pfam/labels/multi_task.Backbone.embeddings = [('target', 'weights')]
pfam/DatasetBuilder.metadata = ('fam_key',)
"""


def make_fake_raw_sequence_dataset(num_examples = 1000):
  voc = vocabulary.alternative
  sampler = vocabulary.Sampler(voc)
  raw_sequences = [voc.decode(s) for s in sampler.sample((num_examples, 128))]
  ds = tf.data.Dataset.from_tensor_slices({
      'sequence': raw_sequences,
      'seq_key': tf.range(num_examples, dtype=tf.int32),
      'fam_key': tf.range(num_examples, 2 * num_examples, dtype=tf.int32),
  })
  return ds


class MultiDatasetBuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()
    gin.parse_config(GIN_CONFIG_MULTI)
    self.addCleanup(mock.patch.stopall)
    self.mock_load1 = mock.patch.object(
        loaders.TFDSLoader, 'load', autospec=True).start()
    self.mock_load2 = mock.patch.object(
        loaders.CSVLoader, 'load', autospec=True).start()

    self.batch = (32, 16)
    self.lengths = (1024, 512)  # from GIN_CONFIG_MULTI.

  def test_transforms_from_gin(self):
    self.mock_load1.return_value = make_fake_sequence_dataset()
    self.mock_load2.return_value = make_fake_raw_sequence_dataset()
    ds_builder = builder.MultiDatasetBuilder()
    ds = ds_builder.make('test', self.batch)
    inputs, y_true, weights, metadata = next(iter(ds))

    self.assertIsInstance(inputs, Sequence)
    for inputs_i, batch_i, length_i in zip(inputs, self.batch, self.lengths):
      self.assertIsInstance(inputs_i, tf.Tensor)
      self.assertEqual(inputs_i.dtype, tf.int32)
      self.assertEqual(inputs_i.shape, (batch_i, length_i))

    self.assertIsInstance(y_true, dict)
    self.assertGreater(len(y_true), 0)
    for y, batch_i, length_i in zip(multi_task.Backbone.unflatten(y_true),
                                    reversed(self.batch),
                                    reversed(self.lengths)):
      self.assertEqual(y.dtype, tf.int32)
      self.assertEqual(y.shape, (batch_i, length_i))

    self.assertIsInstance(weights, dict)
    self.assertGreater(len(weights), 0)
    for w, batch_i, length_i in zip(multi_task.Backbone.unflatten(weights),
                                    reversed(self.batch),
                                    reversed(self.lengths)):
      self.assertEqual(w.dtype, tf.float32)
      self.assertEqual(w.shape, (batch_i, length_i))

    self.assertIn('seq_key', metadata)
    self.assertIsInstance(metadata['seq_key'], tf.Tensor)
    self.assertEqual(metadata['seq_key'].dtype, tf.int32)
    self.assertEqual(metadata['seq_key'].shape, [self.batch[0]])  # from gin.
    self.assertIn('fam_key', metadata)
    self.assertIsInstance(metadata['fam_key'], tf.Tensor)
    self.assertEqual(metadata['fam_key'].dtype, tf.int32)
    self.assertEqual(metadata['fam_key'].shape, [self.batch[1]])  # from gin.


if __name__ == '__main__':
  tf.test.main()
