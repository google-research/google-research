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

"""Self-supervised model for contrastive learning task."""

import os

import tensorflow as tf

from cola import constants
from cola import data
from cola import network


class ContrastiveModel:
  """Provides functionality for self-supervised constrastive learning model."""

  def __init__(self,
               strategy,
               ssl_dataset_name,
               ds_dataset_name,
               model_path,
               experiment_id,
               batch_size,
               epochs, learning_rate,
               embedding_dim,
               temperature,
               similarity_type,
               pooling_type,
               noise,
               steps_per_epoch = 1000):
    """Initializes a contrastive model object."""

    self._strategy = strategy
    self._ssl_dataset_name = ssl_dataset_name
    self._ds_dataset_name = ds_dataset_name
    self._model_path = model_path
    self._experiment_id = experiment_id

    self._batch_size = batch_size
    self._epochs = epochs
    self._learning_rate = learning_rate
    self._temperature = temperature
    self._embedding_dim = embedding_dim
    self._similarity_type = similarity_type
    self._pooling_type = pooling_type
    self._noise = noise

    self._steps_per_epoch = steps_per_epoch
    self._shuffle_buffer = 1000
    self._n_frames = None
    self._n_bands = 64
    self._n_channels = 1
    self._input_shape = (-1, self._n_frames, self._n_bands, self._n_channels)

  def _prepare_example(self, example):
    """Creates an example (anchor-positive) for instance discrimination."""
    x = tf.math.l2_normalize(example["audio"], epsilon=1e-9)

    waveform_a = data.extract_window(x)
    mels_a = data.extract_log_mel_spectrogram(waveform_a)
    frames_anchors = mels_a[Ellipsis, tf.newaxis]

    waveform_p = data.extract_window(x)
    waveform_p = waveform_p + (
        self._noise * tf.random.normal(tf.shape(waveform_p)))
    mels_p = data.extract_log_mel_spectrogram(waveform_p)
    frames_positives = mels_p[Ellipsis, tf.newaxis]

    return frames_anchors, frames_positives

  def _get_ssl_task_data(self):
    """Prepares a dataset for contrastive self-supervised task."""
    ds = data.get_self_supervised_data(self._ssl_dataset_name).repeat()
    ds = ds.shuffle(self._shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.map(
        self._prepare_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(self._batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def train(self):
    """Trains a self-supervised model for contrastive learning."""

    train_dataset = self._get_ssl_task_data()
    train_dataset = self._strategy.experimental_distribute_dataset(
        train_dataset)

    with self._strategy.scope():
      contrastive_network = network.get_contrastive_network(
          embedding_dim=self._embedding_dim,
          temperature=self._temperature,
          pooling_type=self._pooling_type,
          similarity_type=self._similarity_type)
      contrastive_network.compile(
          optimizer=tf.keras.optimizers.Adam(self._learning_rate),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    ssl_model_dir = f"{self._ssl_dataset_name.value}/{self._experiment_id}/"
    ckpt_path = os.path.join(self._model_path, ssl_model_dir, "ckpt_{epoch}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_weights_only=True, monitor="loss")

    backup_path = os.path.join(self._model_path, ssl_model_dir, "backup")
    backandrestore_callback = tf.keras.callbacks.experimental.BackupAndRestore(
        backup_dir=backup_path)

    log_dir = os.path.join(self._model_path, "log", self._experiment_id)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    contrastive_network.fit(
        train_dataset,
        epochs=self._epochs,
        steps_per_epoch=self._steps_per_epoch,
        verbose=2,
        callbacks=[
            model_checkpoint_callback,
            backandrestore_callback,
            tensorboard_callback,
        ])
