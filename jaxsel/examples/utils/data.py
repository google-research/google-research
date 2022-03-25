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

"""Contains utilities for dataloading."""

from typing import Tuple, Sequence

import jax.numpy as jnp
import numpy as np
from scipy import ndimage as ndi
import tensorflow as tf
import tensorflow_datasets as tfds

from jaxsel._src import image_graph
from jaxsel.examples import pathfinder_data


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def load_mnist(
    batch_size = 64
):
  """Load MNIST train and test datasets into memory.

  Taken from https://github.com/google/flax/blob/main/examples/mnist/train.py.

  Args:
    batch_size: batch size for both train and test.

  Returns:
    train_dataset, test_dataset, image_shape, num_classes
  """
  train_dataset = tfds.load('mnist', split='train', as_supervised=True)
  test_dataset = tfds.load('mnist', split='test', as_supervised=True)

  train_dataset = train_dataset.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset.cache()
  test_dataset.cache()

  train_dataset = train_dataset.shuffle(
      60_000, seed=0, reshuffle_each_iteration=True)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
  return train_dataset, test_dataset, (28, 28), 10


def load_pathfinder(
    batch_size = 64,
    resolution = 32,
    difficulty = 'easy',
    overfit = False
):
  """Loads the pathfinder data.

  Args:
    batch_size: batch size for train, test and val datasets.
    resolution: resolution of the task. Can be 32, 64 or 128.
    difficulty: difficulty of the task, defined by the number of distractor
      paths. Must be in ['easy', 'intermediate', 'hard'].
    overfit: if True, the datasets are all the same: first 2 samples of the
      validation dataset.

  Returns:
    train_dataset, val_dataset, test_dataset, image_shape, num_classes
  """
  (train_dataset, val_dataset, test_dataset, num_classes, vocab_size,
   image_shape) = pathfinder_data.load(
       n_devices=1,
       batch_size=batch_size,
       resolution=resolution,
       normalize=True,  # Normalize to 0, 1
       difficulty=difficulty)

  del vocab_size

  if overfit:
    # Doesn't use batch_size in this case
    n_overfit = 8
    train_dataset = val_dataset.unbatch().take(n_overfit).batch(n_overfit)
    val_dataset = train_dataset
    test_dataset = train_dataset

  # Make datasets returns tuples of images, labels
  def tupleize(datapoint):
    return tf.cast(datapoint['inputs'], tf.float32), datapoint['targets']

  train_dataset = train_dataset.map(
      tupleize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val_dataset = val_dataset.map(
      tupleize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.map(
      tupleize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return train_dataset, val_dataset, test_dataset, image_shape, num_classes


# TODO(gnegiar): Map this on the dataset, and cache it.
def make_graph_mnist(
    image, patch_size, bins = (0., .3, 1.)
):
  """Makes a graph object to hold an MNIST sample.

  Args:
    image: Should be squeezable to a 2d array
    patch_size: size of patches for node features.
    bins: Used for binning the pixel values. The highest bin must be greater
      than the highest value in image.

  Returns:
    graph representing the image.
  """
  return image_graph.ImageGraph.create(
      # The threshold value .3 was selected to keep information
      # while not introducing noise
      jnp.digitize(image, bins).squeeze(),
      get_start_pixel_fn=lambda _: (14, 14),  # start in the center
      num_colors=len(bins),  # number of bins + 'out of bounds' pixel
      patch_size=patch_size)


# TODO(gnegiar): Map this on the dataset, and cache it.
def make_graph_pathfinder(
    image,
    patch_size,
    bins,
):
  """Makes a graph holding a pathfinder image.

  Args:
    image: Should be squeezable to a 2d array
    patch_size: size of patches for node features.
    bins: Used for binning the pixel values. The highest bin must be greater
      than the highest value in image.

  Returns:
    graph representing the image.
  """

  # TODO(gnegiar): Allow multiple start nodes.
  # The threshold value .3 was selected to keep information
  # while not introducing noise
  def _get_start_pixel_fn(image, thresh=.3):
    """Detects a probable start point in a Pathfinder image example."""
    thresh_image = np.where(image > thresh, 1, 0)
    distance = ndi.distance_transform_edt(thresh_image)
    idx = distance.argmax()
    coords = np.unravel_index(idx, thresh_image.shape)
    return coords

  # TODO(gnegiar): Allow continuous features in models.
  return image_graph.ImageGraph.create(
      jnp.digitize(image, bins).squeeze(),
      get_start_pixel_fn=_get_start_pixel_fn,
      num_colors=len(bins),  # number of bins + 'out of bounds' pixel
      patch_size=patch_size)
