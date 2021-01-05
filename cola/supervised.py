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

"""Supervised model for fine-tuning, random encoder and from scratch training."""
import functools
import os
from typing import Tuple

from absl import logging
import tensorflow as tf

from cola import constants
from cola import data
from cola import network


class SupervisedModule:
  """Provides functionality for self-supervised source separation model."""

  def __init__(self,
               ssl_dataset_name,
               ds_dataset_name,
               model_path,
               experiment_id,
               batch_size = 64,
               epochs = 100,
               learning_rate = 0.001,
               n_frames = 98,
               n_bands = 64,
               n_channels = 1):
    """Initializes a supervised model object."""

    self._ssl_dataset_name = ssl_dataset_name
    self._ds_dataset_name = ds_dataset_name
    self._model_path = model_path
    self._experiment_id = experiment_id

    self._batch_size = batch_size
    self._epochs = epochs
    self._learning_rate = learning_rate
    self._num_classes = None  # set by _prepare_downstream_task_data
    self._n_frames = n_frames
    self._n_bands = n_bands
    self._n_channels = n_channels
    self._shuffle_buffer = 1000

  def _prepare_standard_example(self, example, is_training):
    """Creates an example for supervised training."""
    x = example["audio"]
    if is_training:
      x = data.extract_window(x)
      x = tf.math.l2_normalize(x, epsilon=1e-9)
    else:
      x = tf.signal.frame(
          x,
          frame_length=self._n_frames * 160,
          frame_step=self._n_frames * 160,
          pad_end=True)
      x = tf.math.l2_normalize(x, axis=-1, epsilon=1e-9)

    x = data.extract_log_mel_spectrogram(x)
    x = x[Ellipsis, tf.newaxis]
    y = example["label"]
    return x, y

  def _prepare_downstream_task_data(self):
    """Get downstream task data."""
    train_data, test_data, self._num_classes = data.get_downstream_dataset(
        self._ds_dataset_name, self._shuffle_buffer)

    train_data = train_data.map(
        functools.partial(self._prepare_standard_example, is_training=True),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            self._batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    test_data = test_data.map(
        functools.partial(self._prepare_standard_example, is_training=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(
            tf.data.experimental.AUTOTUNE)

    return train_data, test_data

  def train_eval(
      self,
      ssl_model_ckpt_id = None,
      load_pretrained = True,
      freeze_encoder = True,
      contrastive_temperature = 0.2,
      contrastive_embedding_dim = 512,
      contrastive_similarity_type = constants
      .SimilarityMeasure.DOT,
      contrastive_pooling_type = "max",
  ):
    """Trains and evaluates a downstream model in any of the below mentioned modes.

       On-top of frozen pre-trained model, fully-supervised, or random fixed
       encoder. All the parameters will only be used when training model on top
       of a pre-trained feature extractor.
    Args:
      ssl_model_ckpt_id: Self-supervised model checkpoint id based on experiment
        and worker id.
      load_pretrained: Boolean to indicate whether to use pre-trained
        self-supervised model for down-stream task.
      freeze_encoder: Boolean to indicate whether to keep the encoder fix or
        train it entirely on down-stream task.
      contrastive_temperature: Temperature value to normalize similarity in
        contrastive model.
      contrastive_embedding_dim: Embedding size of last layer in contrastive
        model.
      contrastive_similarity_type: Similarity measure in contrastive model.
      contrastive_pooling_type: Pooling to use for efficient net.
    """

    train_data, test_data = self._prepare_downstream_task_data()

    if load_pretrained:
      if ssl_model_ckpt_id is None:
        raise ValueError(
            "Self-supervised checkpoint id must not be None for loading pretrained model."
        )

      ssl_model_dir = "{0}/{1}".format(self._ssl_dataset_name.value,
                                       ssl_model_ckpt_id)
      ckpt_path = os.path.join(self._model_path, ssl_model_dir)

      ssl_network = network.get_contrastive_network(
          embedding_dim=contrastive_embedding_dim,
          temperature=contrastive_temperature,
          similarity_type=contrastive_similarity_type,
          pooling_type=contrastive_pooling_type)
      ssl_network.compile(
          optimizer=tf.keras.optimizers.Adam(self._learning_rate),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
      ssl_network.load_weights(
          tf.train.latest_checkpoint(ckpt_path)).expect_partial()
      encoder = ssl_network.embedding_model.get_layer("encoder")
    else:
      encoder = network.get_efficient_net_encoder(
          input_shape=(None, self._n_bands, self._n_channels),
          pooling=contrastive_pooling_type)

    inputs = tf.keras.layers.Input(
        shape=(None, self._n_bands, self._n_channels))
    x = encoder(inputs)
    outputs = tf.keras.layers.Dense(self._num_classes, activation=None)(x)
    model = tf.keras.Model(inputs, outputs)
    if freeze_encoder:
      model.get_layer("encoder").trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(self._learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model.summary()

    ds_model_dir = f"{self._ds_dataset_name.value}/{self._experiment_id}/"
    backup_path = os.path.join(self._model_path, ds_model_dir, "backup")
    backandrestore_callback = tf.keras.callbacks.experimental.BackupAndRestore(
        backup_dir=backup_path)

    model.fit(
        train_data,
        epochs=self._epochs,
        verbose=2,
        callbacks=[backandrestore_callback])

    time_distributed_input = tf.keras.layers.Input(
        shape=(None, None, self._n_bands, 1))
    x = tf.keras.layers.TimeDistributed(model)(time_distributed_input)
    time_averaged_output = tf.reduce_mean(x, axis=1)
    time_distributed_model = tf.keras.Model(time_distributed_input,
                                            time_averaged_output)
    time_distributed_model.compile(
        optimizer=tf.keras.optimizers.Adam(self._learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    test_loss, test_acc = time_distributed_model.evaluate(test_data, verbose=2)

    logging.info("Final test loss: %f", test_loss)
    logging.info("Final test accuracy: %f", test_acc)
