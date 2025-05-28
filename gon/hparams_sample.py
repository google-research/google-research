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

"""Code to generate hparam-metric pairs for the hyperparameter optimization experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_DATASET = flags.DEFINE_string(
    "dataset", "mnist",
    "cifar10 / cifar100 / mnist / fashion_mnist / svhn_cropped")
_HPARAMS = flags.DEFINE_string(
    "hparams", "",
    "use the following format: conv_units1;conv_units2;conv_units3;dense_units1;dense_units2;kernel_width;pool_width;epochs;method"
)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def main(_):
  tfds.disable_progress_bar()

  if _DATASET == "mnist" or _DATASET == "fashion_mnist":
    input_shape = (28, 28, 1)
    output_size = 10
  elif _DATASET == "cifar10" or _DATASET == "svhn_cropped":
    input_shape = (32, 32, 3)
    output_size = 10
  elif _DATASET == "cifar100":
    input_shape = (32, 32, 3)
    output_size = 100

  if _HPARAMS:
    ds_train, ds_test = tfds.load(
        _DATASET,
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True)
  else:
    ds_train, ds_test = tfds.load(
        _DATASET,
        split=["train[0%:90%]", "train[90%:100%]"],
        shuffle_files=True,
        as_supervised=True)
  ds_train = ds_train.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.cache()
  ds_train = ds_train.batch(128)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.batch(128)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  if _HPARAMS:
    hparams = _HPARAMS.split(";")
    conv_units1 = int(hparams[0])
    conv_units2 = int(hparams[1])
    conv_units3 = int(hparams[2])
    dense_units1 = int(hparams[3])
    dense_units2 = int(hparams[4])
    kernel_width = int(hparams[5])
    pool_width = int(hparams[6])
    epochs = int(hparams[7])
  else:
    conv_units1 = int(np.round(random.uniform(8, 512)))
    conv_units2 = int(np.round(random.uniform(8, 512)))
    conv_units3 = int(np.round(random.uniform(8, 512)))
    dense_units1 = int(np.round(random.uniform(8, 512)))
    dense_units2 = int(np.round(random.uniform(8, 512)))
    kernel_width = int(np.round(random.uniform(2, 6)))
    pool_width = int(np.round(random.uniform(2, 6)))
    epochs = int(np.round(random.uniform(1, 25)))

  model = tf.keras.models.Sequential()
  model.add(
      tf.keras.layers.Conv2D(
          conv_units1, (kernel_width, kernel_width),
          activation="relu",
          padding="same",
          input_shape=input_shape))
  model.add(
      tf.keras.layers.MaxPooling2D((pool_width, pool_width), padding="same"))
  model.add(
      tf.keras.layers.Conv2D(
          conv_units2, (kernel_width, kernel_width),
          padding="same",
          activation="relu"))
  model.add(
      tf.keras.layers.MaxPooling2D((pool_width, pool_width), padding="same"))
  model.add(
      tf.keras.layers.Conv2D(
          conv_units3, (kernel_width, kernel_width),
          padding="same",
          activation="relu"))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(dense_units1, activation="relu"))
  model.add(tf.keras.layers.Dense(dense_units2, activation="relu"))
  model.add(tf.keras.layers.Dense(output_size, activation="softmax"))

  model.compile(
      loss="sparse_categorical_crossentropy",
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"],
  )
  history = model.fit(
      ds_train, epochs=epochs, validation_data=ds_test, verbose=0)

  print("[metric] conv_units1=" + str(conv_units1))
  print("[metric] conv_units2=" + str(conv_units2))
  print("[metric] conv_units3=" + str(conv_units3))
  print("[metric] dense_units1=" + str(dense_units1))
  print("[metric] dense_units2=" + str(dense_units2))
  print("[metric] kernel_width=" + str(kernel_width))
  print("[metric] pool_width=" + str(pool_width))
  print("[metric] epochs=" + str(epochs))
  print("[metric] val_accuracy=" + str(history.history["val_accuracy"][-1]))
  print(history.history)


if __name__ == "__main__":
  app.run(main)
