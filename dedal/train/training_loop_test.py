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

"""Tests for training_loop."""

import functools
import os
import tempfile
from unittest import mock

import gin
import tensorflow as tf

from dedal import multi_task
from dedal import vocabulary
from dedal.data import builder
from dedal.data import loaders
from dedal.models import aligners
from dedal.models import dedal
from dedal.models import encoders
from dedal.models import homology
from dedal.train import logger
from dedal.train import losses
from dedal.train import training_loop

CONFIG_FOLDER = 'configs/'
open_fn = open



def parse_gin_config():
  filename = os.path.join(CONFIG_FOLDER, 'debug.gin')
  with open_fn(filename, 'rt') as file_handle:
    gin.parse_config(file_handle)


def get_strategy():
  """Get the strategy corresponding to the current setup."""
  tpus = tf.config.experimental.list_logical_devices(device_type='TPU')
  use_tpu = bool(tpus)

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if len(gpus) >= 1:
    tf.config.set_logical_device_configuration(
        gpus[0], [
            tf.config.LogicalDeviceConfiguration(1024),
            tf.config.LogicalDeviceConfiguration(1024)
        ])
  return training_loop.get_strategy(use_tpu)


def make_fake_dataset(num_examples = 1000):
  voc = vocabulary.proteins
  sampler = vocabulary.Sampler(voc)
  ds = tf.data.Dataset.from_tensor_slices(sampler.sample((num_examples, 128)))
  return ds.map(lambda x: {'sequence': x})


def make_fake_homology_dataset(num_examples = 1000, seq_len = 128):
  voc = vocabulary.proteins
  sampler = vocabulary.Sampler(voc)
  return tf.data.Dataset.from_tensor_slices({
      'sequence': sampler.sample((num_examples, seq_len)),
      'target': tf.random.uniform(shape=(num_examples,)) > 0.8,
      'weights': tf.ones(shape=(num_examples, seq_len), dtype=tf.float32),
  })


class TrainingLoopTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()
    parse_gin_config()
    self.addCleanup(mock.patch.stopall)
    self._mock_load = mock.patch.object(
        loaders.TFDSLoader, 'load', autospec=True).start()
    self._mock_load.return_value = make_fake_dataset(1000)

    workdir = tempfile.mkdtemp()
    strategy = get_strategy()
    self._loop = training_loop.TrainingLoop(
        workdir=workdir, strategy=strategy, graph_mode=True)

  def _make_loop_with_reference(self):
    """Creates a training loop for alignments with self._loop as reference."""
    seq_len = gin.query_parameter('%SEQUENCE_LENGTH')
    model_cls = functools.partial(
        dedal.Dedal,
        encoder_cls=functools.partial(
            encoders.TransformerEncoder,
            emb_dim=48,
            num_layers=1,
            num_heads=2,
            mlp_dim=3 * seq_len,
            max_len=seq_len),
        aligner_cls=aligners.SoftAligner,
        heads_cls=multi_task.Backbone(
            embeddings=[],
            alignments=[homology.UncorrectedLogits]
            ),
        )
    workdir2 = tempfile.mkdtemp()
    ds_builder = builder.DatasetBuilder(
        labels=multi_task.Backbone(alignments=[('target', 'weights')]))
    return training_loop.TrainingLoop(
        workdir=workdir2, strategy=self._loop.strategy,
        dataset_builder=ds_builder,
        logger_cls=functools.partial(
            logger.Logger,
            scalars=multi_task.Backbone(
                alignments=[[tf.keras.metrics.BinaryAccuracy]]),
            every=5),
        loss_fn=losses.MultiTaskLoss(losses=multi_task.Backbone(
            alignments=[tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)])),
        optimizer_cls=self._loop._optimizer_cls,
        batch_size=self._loop._batch_size,
        model_cls=model_cls,
        num_steps=self._loop._num_steps,
        reference_workdir=self._loop._workdir,
        num_reference_steps=self._loop._num_steps)

  def test_train_step(self):
    # First trains the model as defined in debug.gin for language model.
    self._loop.train()
    events = tf.io.gfile.glob(os.path.join(self._loop._workdir, 'train', '*'))
    self.assertLen(events, 1)
    self.assertIn('tfevents', events[0])

    ckpts = tf.io.gfile.glob(
        os.path.join(self._loop._workdir, 'checkpoints', '*.index'))
    num_expected_checkpoints = (
        self._loop._num_steps // self._loop._checkpointer._save_every)
    self.assertLen(ckpts, num_expected_checkpoints)

  def test_transfer(self):
    """First train a model then transfer its weights to another one."""
    self._loop.train()
    loop = self._make_loop_with_reference()
    self.assertIsNotNone(loop._reference_ckpt)
    self.assertEqual(loop._reference_step, 0)
    seq_len = gin.query_parameter('%SEQUENCE_LENGTH')
    loop.may_transfer(tf.zeros((loop._batch_size, seq_len), dtype=tf.int32),
                      freeze=True)
    for i, weights in enumerate(self._loop.model.encoder.weights):
      self.assertAllEqual(weights, loop.model.encoder.weights[i])
    self.assertEmpty(loop.model.encoder.trainable_weights)

  def test_pretrain(self):
    """First train a model then transfer and train another one."""
    self._loop.train()

    loop2 = self._make_loop_with_reference()
    with mock.patch.object(
        loop2._dataset_builder, 'prepare', autospec=True) as ds_builder_mock:
      ds_builder_mock.return_value = make_fake_homology_dataset(1000)
      loop2.train()

    events = tf.io.gfile.glob(os.path.join(loop2._workdir, 'train', '*'))
    self.assertLen(events, 1)
    self.assertIn('tfevents', events[0])

    ckpts = tf.io.gfile.glob(
        os.path.join(loop2._workdir, 'checkpoints', '*.index'))
    num_expected_checkpoints = (
        loop2._num_steps // loop2._checkpointer._save_every)
    self.assertLen(ckpts, num_expected_checkpoints)

  def test_evaluate(self):
    self._loop.train()
    events = tf.io.gfile.glob(os.path.join(self._loop._workdir, 'test', '*'))
    self.assertLen(events, 0)
    with self._loop.strategy.scope():
      self._loop._step.assign(0)  # Back to step 0
    self._loop.evaluate()
    events = tf.io.gfile.glob(os.path.join(self._loop._workdir, 'test', '*'))
    self.assertLen(events, 1)
    self.assertIn('tfevents', events[0])

  def test_serial_evaluate(self):
    self._loop.train()
    splits = ('valid', 'test')
    self._loop._dataset_builder.split = splits
    for split in splits:
      events = tf.io.gfile.glob(os.path.join(self._loop._workdir, split, '*'))
      self.assertLen(events, 0)
    with self._loop.strategy.scope():
      self._loop._step.assign(0)  # Back to step 0
    self._loop.evaluate()
    for split in splits:
      events = tf.io.gfile.glob(os.path.join(self._loop._workdir, split, '*'))
      self.assertLen(events, 1)
      self.assertIn('tfevents', events[0])

  def test_eval_on_train(self):
    # First trains and saves a model.
    self._loop.train()
    workdir = self.create_tempdir()
    # Make sure we use the 'train' split.
    self._loop._dataset_builder.split = 'train'
    loop = training_loop.TrainingLoop(
        workdir=workdir,
        strategy=self._loop.strategy,
        dataset_builder=self._loop._dataset_builder,
        logger_cls=self._loop._logger_cls,
        model_cls=self._loop._model_cls,
        loss_fn=self._loop._loss_fn,
        optimizer_cls=None,
        batch_size=self._loop._batch_size,
        num_steps=self._loop._num_steps,
        graph_mode=self._loop._graph_mode,
        reference_workdir=self._loop._workdir,  # reads model from here.
        num_reference_steps=self._loop._num_steps)
    loop.evaluate()

    events = tf.io.gfile.glob(os.path.join(workdir, 'test', '*.tfevent*'))
    self.assertEmpty(events)
    events = tf.io.gfile.glob(os.path.join(workdir, 'train', '*.tfevent*'))
    self.assertNotEmpty(events)

    # Make sure we have the same weights
    for i, weights in enumerate(self._loop.model.encoder.weights):
      self.assertAllEqual(weights, loop.model.encoder.weights[i])

  def test_downstream(self):
    self._loop.train()
    loop = self._make_loop_with_reference()
    with mock.patch.object(
        loop._dataset_builder, 'prepare', autospec=True) as ds_builder_mock:
      ds_builder_mock.return_value = make_fake_homology_dataset(1000)
      loop.downstream()

    workdir = loop._workdir
    events = tf.io.gfile.glob(os.path.join(workdir))
    self.assertNotEmpty(events)
    events = tf.io.gfile.glob(os.path.join(workdir, 'train', '*.tfevent*'))
    self.assertNotEmpty(events)

    for i, weights in enumerate(self._loop.model.encoder.weights):
      self.assertAllEqual(weights, loop.model.encoder.weights[i])


if __name__ == '__main__':
  tf.test.main()
