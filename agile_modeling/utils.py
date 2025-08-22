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

# pylint: disable-all
import io
from multiprocessing import Pool
from typing import Any, Optional, Sequence
import urllib
import urllib.request

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def download_image(url):
  """Downloads an image.

  Args:
    url: The image URL.

  Returns:
    A PIL image object, or None if an error occured.
  """
  url_https = url.replace('http://', 'https://')
  try:
    req = urllib.request.Request(url_https)
    response = urllib.request.urlopen(req, timeout=1)
    image = Image.open(io.BytesIO(response.read()))
    scale = 400 / image.size[1]
    image = image.resize((int(image.size[0] * scale), 400), Image.LANCZOS)
    return image
  except:  # pylint: disable=bare-except
    return None


def download_images_parallel(urls):
  """Checks whether a list of urls is valid.

  Args:
    urls: a list of urls.

  Returns:
    A list of images corresponding to the urls.
  """
  pool = Pool()
  return pool.map(download_image, urls)


def check_image_exists(url):
  """Checks if an image at some URL really exists. Converts http to https first.

  Args:
    url: The image URL

  Returns:
    True if the image exists, otherwise False
  """
  url_https = url.replace('http://', 'https://')
  try:
    req = urllib.request.Request(url_https)
    response = urllib.request.urlopen(req, timeout=1)
    _ = Image.open(io.BytesIO(response.read()))
    return response.code in range(200, 209)
  except:  # pylint: disable=bare-except
    return False


def check_images_exist_parallel(urls):
  """Checks whether a list of urls is valid.

  Args:
    urls: a list of urls.

  Returns:
    A list of bools corresponding to the order of urls.
  """
  pool = Pool()
  return pool.map(check_image_exists, urls)


def create_classifier(
    layer_dims = [32],
    dropout = 0.2,
    lr = 0.0003,
    weight_decay = 1e-4,
    feature_dim = 512,
    model_output_logits = True,
):
  """Creates a small MLP classifier on top of pre-computed embeddings.

  Args:
    layer_dims: Adds a linear layer + ReLU for each element in the list. The
      size of the linear layer is layer_dims[i].
    dropout: Dropout percentage.
    lr: learning rate.
    weight_decay: weight decay.
    feature_dim: The input feature embedding size.
    model_output_logits: If true, outputs logits.

  Returns:
    The MLP classifier.
  """

  layer_list = []
  for i, dim in enumerate(layer_dims):
    layer_list.append(
        layers.Dense(
            dim,
            activation='relu',
            name=f'layer_{i+1}',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        )
    )
    layer_list.append(layers.Dropout(dropout, name=f'dropout_{i+1}'))

  model = keras.Sequential([
      layers.InputLayer(input_shape=(feature_dim,)),
      *layer_list,
      layers.Dense(
          1,
          name='out',
          activation=None if model_output_logits else 'sigmoid',
          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
      ),
  ])

  metrics = [
      tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'),
      tf.keras.metrics.AUC(curve='ROC', name='auc_roc'),
      tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
      tf.keras.metrics.RecallAtPrecision(precision=0.7, name='recall@prec0.7'),
      tf.keras.metrics.RecallAtPrecision(precision=0.8, name='recall@prec0.8'),
      tf.keras.metrics.RecallAtPrecision(precision=0.9, name='recall@prec0.9'),
      tf.keras.metrics.TruePositives(),
      tf.keras.metrics.FalsePositives(),
      tf.keras.metrics.TrueNegatives(),
      tf.keras.metrics.FalseNegatives(),
      tf.keras.metrics.PrecisionAtRecall(
          recall=0.7, name='precision@recall0.7'
      ),
      tf.keras.metrics.PrecisionAtRecall(
          recall=0.8, name='precision@recall0.8'
      ),
      tf.keras.metrics.PrecisionAtRecall(
          recall=0.9, name='precision@recall0.9'
      ),
  ]

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=model_output_logits),
      metrics=metrics,
  )

  return model


def train_model(
    model,
    x,
    y,
    validation_data = None,
    verbose = 1,
    batch_size = 128,
    epochs = 16,
    step_multiplier = 4,
    patience = 10,
):
  """Trains the model.

  Args:
    model: The model to train.
    x: The training features.
    y: The training labels.
    validation_data: Validation data, as needed by Keras model.fit()
    verbose: The level of print verbosity.
    batch_size: The training batch size
    epochs: The number of epochs.
    step_multiplier: A scaling of images per epoch.
    patience: Early stopping patience.

  Returns:
    The history object returned by Keras's model.fit().
  """
  callbacks = []
  if validation_data:
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc_pr',
            min_delta=0,
            patience=patience,
            verbose=0,
            mode='max',
            baseline=None,
            restore_best_weights=True,
        )
    )

  history = model.fit(
      x,
      y,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=validation_data,
      verbose=verbose,
      steps_per_epoch=len(x) // (batch_size * step_multiplier),
      shuffle=True,
      callbacks=callbacks,
  )
  return history
