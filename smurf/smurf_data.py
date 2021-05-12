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

"""This is a library for data related functions.

This libary contains two functions, make_train_iterator for generating a
training data iterator from multiple sources of different formats and
make_eval_function for creating an evaluation function that evaluates on
data from multiple sources of different formats.
"""

# pylint:skip-file
from functools import partial

import tensorflow as tf

from smurf import smurf_augmentation
from smurf.data import generic_flow_dataset as flow_dataset
from smurf.data import kitti
from smurf.data import sintel
from smurf.data import smurf_multiframe_dataset
from smurf.data import spoof_dataset


def make_train_dataset(
    train_on,
    height,
    width,
    shuffle_buffer_size,
    batch_size,
    seq_len,
    crop_instead_of_resize=False,
    apply_augmentation=True,
    include_ground_truth=False,
    resize_gt_flow=True,
    seed=41,
    mode='train',
    return_full_scale=True,
):
  """Build joint training dataset for all data in train_on.

  Args:
    train_on: string of the format 'format0:path0;format1:path1', e.g.
       'kitti:/tmp/...'.
    height: int, height to which the images will be resized or cropped.
    width: int, width to which the images will be resized or cropped.
    shuffle_buffer_size: int, size that will be used for the shuffle buffer.
    batch_size: int, batch size for the iterator.
    seq_len: int, number of frames per sequences (at the moment this should
      always be 2)
    crop_instead_of_resize: bool, indicates if cropping should be used instead
      of resizing
    apply_augmentation: bool, indicates if geometric and photometric data
      augmentation shall be activated (paramaters are gin configurable)
    include_ground_truth: bool, indicates if ground truth flow should be
      included.
    resize_gt_flow: bool, indicates if ground truth flow should be resized (only
      important if resizing and supervised training is used)
    seed: A seed for a random number generator, controls shuffling of data.
    mode: str, the mode to pass to the data loader. defaults to 'train'
    return_full_scale: bool, whether or not to include the full size, uncropped
      images in the data dictionary.

  Returns:
    data: A tf.data.Iterator that produces batches of data dictionaries.
  """

  train_datasets = []
  # Split strings according to pattern "format0:path0;format1:path1".
  for format_and_path in train_on.split(';'):

    data_format, path = format_and_path.split(':')

    if include_ground_truth:
      mode += '-supervised'

    # Add a dataset based on format and path.
    if 'spoof' in data_format:
      dataset = spoof_dataset.make_dataset(
          path,
          mode=mode,
          seq_len=seq_len,
          shuffle_buffer_size=shuffle_buffer_size,
          height=None if crop_instead_of_resize else height,
          width=None if crop_instead_of_resize else width,
          resize_gt_flow=resize_gt_flow,
          seed=seed,
      )
    elif 'multiframe' in data_format:  # Multiframe data.
      dataset_manager = smurf_multiframe_dataset.SmurfMultiframe()
      dataset = dataset_manager.make_dataset(
          path,
          mode=mode,
          seq_len=seq_len,
          shuffle_buffer_size=shuffle_buffer_size,
          height=None if crop_instead_of_resize else height,
          width=None if crop_instead_of_resize else width,
          resize_gt_flow=resize_gt_flow,
          seed=seed,
      )
    elif 'kitti' in data_format:
      dataset = kitti.make_dataset(
          path,
          mode=mode,
          seq_len=seq_len,
          shuffle_buffer_size=shuffle_buffer_size,
          height=None if crop_instead_of_resize else height,
          width=None if crop_instead_of_resize else width,
          resize_gt_flow=resize_gt_flow,
          seed=seed,
      )
    elif 'chairs' in data_format:
      dataset = flow_dataset.make_dataset(
          path,
          mode=mode,
          seq_len=seq_len,
          shuffle_buffer_size=shuffle_buffer_size,
          height=None if crop_instead_of_resize else height,
          width=None if crop_instead_of_resize else width,
          resize_gt_flow=resize_gt_flow,
          gt_flow_shape=[384, 512, 2],
          seed=seed,
      )
    elif 'sintel' in data_format:
      dataset = sintel.make_dataset(
          path,
          mode=mode,
          seq_len=seq_len,
          shuffle_buffer_size=shuffle_buffer_size,
          height=None if crop_instead_of_resize else height,
          width=None if crop_instead_of_resize else width,
          resize_gt_flow=resize_gt_flow,
          seed=seed,
      )
    else:
      print('Unknown data format "{}"'.format(data_format))
      continue
    train_datasets.append(dataset)

  augmentation_fn = partial(
      smurf_augmentation.apply_augmentation,
      crop_height=height,
      crop_width=width,
      return_full_scale=return_full_scale)

  # After loading and augmentation the data can have unknown shape.
  # The function below ensures that all data has the proper shape.
  def _ensure_shapes():
    # shape of the data
    flow_height = height if resize_gt_flow else None
    flow_width = width if resize_gt_flow else None
    shapes = {
        'images': (batch_size, seq_len, height, width, 3),
        'flow': (batch_size, flow_height, flow_width, 2),
        'flow_valid': (batch_size, flow_height, flow_width, 1),
        'occlusions': (batch_size, height, width, 1),
    }
    def check_data(data):
      output = {}
      for key, val in data.items():
        if key in shapes:
          val = tf.ensure_shape(val, shapes[key])
        output[key] = val
      return output
    return check_data

  choice_dataset = tf.data.Dataset.range(len(train_datasets)).repeat()
  train_ds = tf.data.experimental.choose_from_datasets(train_datasets,
                                                       choice_dataset)

  if apply_augmentation:
    train_ds = train_ds.map(augmentation_fn)

  train_ds = train_ds.batch(batch_size, drop_remainder=True)
  train_ds = train_ds.prefetch(1)
  train_ds = train_ds.map(_ensure_shapes())
  return train_ds


def make_eval_function(eval_on, height, width, progress_bar, plot_dir,
                       num_plots, weights=None):
  """Build an evaluation function for smurf.

  Args:
    eval_on: string of the format 'format0:path0;format1:path1', e.g.
       'kitti:/tmp/...'.
    height: int, the height to which the images should be resized for inference.
    width: int, the width to which the images should be resized for inference.
    progress_bar: boolean, flag to indicate whether the function should print a
      progress_bar during evaluaton.
    plot_dir: string, optional path to a directory in which plots are saved (if
      num_plots > 0).
    num_plots: int, maximum number of qualitative results to plot for the
      evaluation.
    weights: dictionary of loss weights for computing loss on the evaluation
      data.
  Returns:
    data: A pair consisting of an evaluation function and a list of strings
    that holds the keys of the evaluation result.
  """
  eval_functions_and_datasets = []
  eval_keys = []
  # Split strings according to pattern "format0:path0;format1:path1".
  for format_and_path in eval_on.split(';'):
    data_format, path = format_and_path.split(':')

    # Add a dataset based on format and path.
    if 'spoof' in data_format:
      dataset = spoof_dataset.make_dataset(path, mode='eval')
      eval_fn = partial(spoof_dataset.evaluate, prefix=data_format)
      eval_keys += spoof_dataset.list_eval_keys(prefix=data_format)
    elif 'kitti' in data_format:
      if 'benchmark' in data_format:
        dataset = kitti.make_dataset(path, mode='test')
        eval_fn = kitti.benchmark
      else:
        dataset = kitti.make_dataset(path, mode='eval')
        eval_fn = partial(kitti.evaluate, prefix=data_format)
        eval_keys += kitti.list_eval_keys(prefix=data_format)
    elif 'chairs' in data_format:
      dataset = flow_dataset.make_dataset(path, mode='eval')
      eval_fn = partial(
          flow_dataset.evaluate,
          prefix=data_format,
          max_num_evals=500,  # We do this to avoid evaluating on 22k samples.
          has_occlusion=False,
          weights=weights)
      eval_keys += flow_dataset.list_eval_keys(prefix=data_format)
    elif 'sintel' in data_format:
      if 'benchmark' in data_format:
        # pylint:disable=g-long-lambda
        # pylint:disable=cell-var-from-loop
        eval_fn = lambda smurf: sintel.benchmark(
            inference_fn=smurf.infer,
            height=height,
            width=width,
            sintel_path=path,
            plot_dir=plot_dir,
            num_plots=num_plots)
        assert len(eval_on.split(
            ';')) == 1, 'Sintel benchmark should be done in isolation.'
        return eval_fn, []
      dataset = sintel.make_dataset(path, mode='eval-occlusion')
      eval_fn = partial(sintel.evaluate, prefix=data_format,
                        weights=weights)
      eval_keys += sintel.list_eval_keys(prefix=data_format)
    else:
      print('Unknown data format "{}"'.format(data_format))
      continue
    dataset = dataset.prefetch(1)
    eval_functions_and_datasets.append((eval_fn, dataset))

  # Make an eval function that aggregates all evaluations.
  def eval_function(smurf):
    result = dict()
    for eval_fn, ds in eval_functions_and_datasets:
      results = eval_fn(
          smurf.infer, ds, height,
          width, progress_bar, plot_dir, num_plots)
      for k, v in results.items():
        result[k] = v
    return result

  return eval_function, eval_keys
