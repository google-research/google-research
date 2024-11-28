# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Load sequence datasets into tf.data.Dataset pipeline."""

import functools
import os
import numpy as np
import tensorflow.compat.v1 as tf

# Data fields used by the model:
REQUIRED_DATA_FIELDS = ['image', 'true_object_pos']


def get_sequence_dataset(data_dir,
                         batch_size,
                         num_timesteps,
                         file_glob='*.npz',
                         random_offset=True,
                         repeat_dataset=True,
                         seed=0):
  """Returns a tf.data.Dataset object for a Numpy image sequence dataset.

  Args:
    data_dir: Directory containing Numpy files where each file contains an image
      sequence and ground-truth object coordinates.
    batch_size: Desired number of sequences per batch in the output dataset.
    num_timesteps: Desired sequence length in the output dataset.
    file_glob: Glob pattern to sub-select files in data_dir.
    random_offset: If True, a random number of frames will be dropped from the
      start of the input sequence before dividing it into chunks of length
      num_timesteps.
    repeat_dataset: If True, output dataset will repeat forever.
    seed: Random seed for shuffling.

  Returns:
    A tf.data.Dataset, or a one-shot iterator of the dataset if return_iterator
    is True.

  Raises:
    RuntimeError: If no data files are found in data_dir.
  """

  # Find files for dataset. Each file contains a sequence of arbitrary length:
  file_glob = file_glob if '.npz' in file_glob else file_glob + '.npz'
  filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, file_glob)))
  if not filenames:
    raise RuntimeError('No data files match {}.'.format(
        os.path.join(data_dir, file_glob)))

  # Deterministic in-place shuffle:
  np.random.RandomState(seed).shuffle(filenames)

  # Create dataset:
  dtypes, pre_chunk_shapes = _read_data_types_and_shapes(filenames)
  dataset = tf.data.Dataset.from_generator(
      lambda: _read_numpy_sequences(filenames), dtypes, pre_chunk_shapes)

  if repeat_dataset:
    dataset = dataset.repeat()

  # Divide sequences into num_timesteps chunks:
  chunk_fn = functools.partial(
      _chunk_sequence, chunk_length=num_timesteps, random_offset=random_offset)
  dataset = dataset.interleave(chunk_fn, cycle_length=batch_size)

  # Format dataset:
  dataset = dataset.shuffle(
      100 * batch_size, seed=seed, reshuffle_each_iteration=True)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(buffer_size=None)  # None = Auto-tune

  # Get shapes after data formatting:
  format_shape = lambda shape: (None,) + tuple(shape.as_list()[1:])
  shapes = {
      key: format_shape(shape) for key, shape in dataset.output_shapes.items()}

  return dataset, shapes


def _read_numpy_sequences(filenames):
  """Generator that reads Numpy files with sequence data from disk into a dict.

  Unreadable files (i.e. files that cause an IOError) will be skipped.

  For traceability, fields containing the filename and frame number will be
  added to the sequence dict.

  Args:
    filenames: List of paths to Numpy files.

  Yields:
    Dict containing Numpy arrays corresponding to the data for one sequence.
  """

  for filename in filenames:
    try:
      with tf.gfile.Open(filename, 'rb') as f:
        sequence_dict = {k: v for k, v in np.load(f).items()}
    except IOError as e:
      print('Caught IOError: "{}". Skipping file {}.'.format(e, filename))

    # Format data:
    sequence_dict = _choose_data_fields(sequence_dict)
    sequence_dict = {
        k: _adjust_precision_for_tf(v) for k, v in sequence_dict.items()}
    sequence_dict['image'] = _format_image_data(sequence_dict['image'])

    # Add filename and frame arrays for traceability:
    num_frames = list(sequence_dict.values())[0].shape[0]
    sequence_dict['frame_ind'] = np.arange(num_frames, dtype=np.int32)
    sequence_dict['filename'] = np.full(num_frames, os.path.basename(filename))

    yield sequence_dict


def _choose_data_fields(data_dict):
  """Returns a new dict containing only fields required by the model."""
  output_dict = {}
  for k in REQUIRED_DATA_FIELDS:
    if k in data_dict:
      output_dict[k] = data_dict[k]
    elif k == 'true_object_pos':
      # Create dummy ground truth if it's not in the dict:
      tf.logging.log_first_n(tf.logging.WARN,
                             'Found no true_object_pos in data, adding dummy.',
                             1)
      num_timesteps = data_dict['image'].shape[0]
      output_dict['true_object_pos'] = np.zeros([num_timesteps, 0, 2])
    else:
      raise ValueError(
          'Required key "{}" is not in the  dict with keys {}.'.format(
              k, list(data_dict.keys())))
  return output_dict


def _adjust_precision_for_tf(array):
  """Adjusts precision for TensorFlow."""
  if array.dtype == np.float64:
    return array.astype(np.float32)
  if array.dtype == np.int64:
    return array.astype(np.int32)
  return array


def _format_image_data(image):
  """Formats the uint8 input image to float32 in the range [-0.5, 0.5]."""
  if not np.issubdtype(image.dtype, np.uint8):
    raise ValueError('Expected image to be of type {}, but got type {}.'.format(
        np.uint8, image.dtype))
  return image.astype(np.float32) / 255.0 - 0.5


def _read_data_types_and_shapes(filenames):
  """Gets dtypes and shapes for all keys in the dataset."""
  sequences = _read_numpy_sequences(filenames)
  sequence = next(sequences)
  sequences.close()
  dtypes = {k: tf.as_dtype(v.dtype) for k, v in sequence.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in sequence.items()}
  return dtypes, shapes


def _chunk_sequence(sequence_dict, chunk_length, random_offset=False):
  """Splits a dict of sequence tensors into a batch of chunks.

  This function does not expect a batch of sequences, but a single sequence.

  Args:
    sequence_dict: dict of tensors with time along the first dimension.
    chunk_length: Size of chunks the sequence will be split into.
    random_offset: Start chunking from a random offset in the sequence,
      enforcing that at least one chunk is generated.

  Returns:
    tf.data.Dataset of sequence chunks.
  """
  length = tf.shape(list(sequence_dict.values())[0])[0]

  if random_offset:
    num_chunks = tf.maximum(1, length // chunk_length - 1)
    output_length = num_chunks * chunk_length
    max_offset = length - output_length
    offset = tf.random_uniform((), 0, max_offset + 1, dtype=tf.int32)
  else:
    num_chunks = length // chunk_length
    output_length = num_chunks * chunk_length
    offset = 0

  chunked = {}
  for key, tensor in sequence_dict.items():
    tensor = tensor[offset:offset + output_length]
    chunked_shape = [num_chunks, chunk_length] + tensor.shape[1:].as_list()
    chunked[key] = tf.reshape(tensor, chunked_shape)

  filename = sequence_dict['filename'][0]
  seed = tf.strings.to_hash_bucket_fast(filename, num_buckets=2**62)
  return tf.data.Dataset.from_tensor_slices(chunked).shuffle(
      tf.cast(length, tf.int64), seed=seed)
