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

"""Utilities for loading the Pathfinder dataset."""

from lra.lra_benchmarks.data import pathfinder
import tensorflow as tf
import tensorflow_datasets as tfds


# Please set this variable to the path for the LRA pathfinder data.
_PATHFINDER_TFDS_PATH = None



AUTOTUNE = tf.data.experimental.AUTOTUNE


def load(n_devices=1,
         batch_size=256,
         resolution=32,
         normalize=False,
         difficulty='easy'):
  """Get Pathfinder dataset splits.

  Args:
    n_devices: Number of devices used. Default: 1
    batch_size: Batch size
    resolution: Resolution of the images. Either 32, 64 or 128.
    normalize: If True, the images have float elements in [0,1].
    difficulty: Controls the number of distractor paths.

  Returns:
    (train_dataset, val_dataset, test_dataset, num_classes,
    vocab_size, input_shape)
  """

  if _PATHFINDER_TFDS_PATH is None:
    raise ValueError(
        'You must set _PATHFINDER_TFDS_PATH above to your pathfinder data path.'
    )

  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  if difficulty not in ['easy', 'intermediate', 'hard']:
    raise ValueError("difficulty must be in ['easy', 'intermediate', 'hard'].")

  if resolution == 32:
    builder = pathfinder.Pathfinder32(data_dir=_PATHFINDER_TFDS_PATH)
    inputs_shape = (32, 32)
  elif resolution == 64:
    builder = pathfinder.Pathfinder64(data_dir=_PATHFINDER_TFDS_PATH)
    inputs_shape = (64, 64)
  elif resolution == 128:
    builder = pathfinder.Pathfinder128(data_dir=_PATHFINDER_TFDS_PATH)
    inputs_shape = (128, 128)
  elif resolution == 256:
    builder = pathfinder.Pathfinder256(data_dir=_PATHFINDER_TFDS_PATH)
    inputs_shape = (256, 256)
  else:
    raise ValueError('Resolution must be in [32, 64, 128, 256].')

  def get_split(difficulty):
    ds = builder.as_dataset(
        split=difficulty, decoders={'image': tfds.decode.SkipDecoding()})

    # Filter out examples with empty images:
    ds = ds.filter(lambda x: tf.strings.length((x['image'])) > 0)

    return ds

  train_dataset = get_split(f'{difficulty}[:80%]')
  val_dataset = get_split(f'{difficulty}[80%:90%]')
  test_dataset = get_split(f'{difficulty}[90%:]')

  def decode(x):
    decoded = {
        'inputs': tf.cast(tf.image.decode_png(x['image']), dtype=tf.int32),
        'targets': x['label']
    }
    if normalize:
      decoded['inputs'] = decoded['inputs'] / 255
    return decoded

  train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

  # TODO(gnegiar): Don't shuffle and batch here.
  # Let the train.py file convert datapoints to graph representation
  # and cache before shuffling and batching.
  train_dataset = train_dataset.shuffle(
      buffer_size=256 * 8, reshuffle_each_iteration=True)

  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

  return train_dataset, val_dataset, test_dataset, 2, 256, inputs_shape
