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

"""Library for loading train and eval data.

This libary contains two functions, make_train_iterator for generating a
training data iterator from multiple sources of different formats and
make_eval_function for creating an evaluation function that evaluates on
data from multiple sources of different formats.
"""

# pylint:disable=g-importing-member
from functools import partial

import tensorflow as tf

from uflow import uflow_augmentation
from uflow.data import generic_flow_dataset as flow_dataset
from uflow.data import kitti
from uflow.data import sintel

# pylint:disable=g-long-lambda


def make_train_iterator(
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
    include_occlusions=False,
    seed=41,
    mode='train',
):
  """Build joint training iterator for all data in train_on.

  Args:
    train_on: string of the format 'format0:path0;format1:path1', e.g.
       'kitti:/usr/local/home/...'.
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
    include_ground_truth: bool, if True, return ground truth optical flow with
      the training images. This only exists for some datasets (Kitti, Sintel).
    resize_gt_flow: bool, indicates if ground truth flow should be resized (only
      important if resizing and supervised training is used)
    include_occlusions: bool, indicates if ground truth occlusions should be
      loaded (currently not supported in combination with augmentation)
    seed: A seed for a random number generator, controls shuffling of data.
    mode: str, will be passed on to the data iterator class. Can be used to
      specify different settings within the data iterator.

  Returns:
    A tf.data.Iterator that produces batches of images of shape [batch
    size, sequence length=3, height, width, channels=3]
  """
  train_datasets = []
  # Split strings according to pattern "format0:path0;format1:path1".
  for format_and_path in train_on.split(';'):

    data_format, path = format_and_path.split(':')

    if include_occlusions:
      mode += '-include-occlusions'

    if include_ground_truth:
      mode += '-supervised'

    if include_occlusions and 'sintel' not in data_format:
      raise ValueError('The parameter include_occlusions is only supported for'
                       'sintel data.')

    if include_ground_truth and ('chairs' not in data_format and
                                 'sintel' not in data_format and
                                 'kitti' not in data_format):
      raise NotImplementedError('The parameter include_ground_truth is only'
                                'supported for flying_chairs, sintel, kitti and'
                                'wod data at the moment.')

    # Add a dataset based on format and path.
    if 'kitti' in data_format:
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

  # prepare augmentation function
  # in case no crop is desired set it to the size images have been resized to
  # This will fail if none or both are specified.
  augmentation_fn = partial(
      uflow_augmentation.apply_augmentation,
      crop_height=height,
      crop_width=width)

  # returns a function to apply ensure_shape on all the available data
  def _ensure_shapes():
    # shape of the data
    imgs_shape = (batch_size, seq_len, height, width, 3)
    if resize_gt_flow:
      flow_shape = (batch_size, height, width, 2)
      valid_shape = (batch_size, height, width, 1)
    else:
      flow_shape = (batch_size, None, None, 2)
      valid_shape = (batch_size, None, None, 1)
    occ_shape = (batch_size, height, width, 1)

    # different cases of data combinations
    if include_ground_truth and apply_augmentation:
      return lambda imgs, imgs_na, flow, valid: (tf.ensure_shape(
          imgs, imgs_shape), {
              'images_without_photo_aug': tf.ensure_shape(imgs_na, imgs_shape),
              'flow_uv': tf.ensure_shape(flow, flow_shape),
              'flow_valid': tf.ensure_shape(valid, valid_shape)
          })
    elif include_ground_truth and include_occlusions:
      return lambda imgs, flow, valid, occ: (tf.ensure_shape(
          imgs, imgs_shape), {
              'flow_uv': tf.ensure_shape(flow, flow_shape),
              'flow_valid': tf.ensure_shape(valid, valid_shape),
              'occlusions': tf.ensure_shape(occ, occ_shape)
          })
    elif include_ground_truth:
      return lambda imgs, flow, valid: (tf.ensure_shape(imgs, imgs_shape), {
          'flow_uv': tf.ensure_shape(flow, flow_shape),
          'flow_valid': tf.ensure_shape(valid, valid_shape)
      })
    elif include_occlusions:
      return lambda imgs, occ: (tf.ensure_shape(imgs, imgs_shape), {
          'occlusions': tf.ensure_shape(occ, occ_shape)
      })
    elif apply_augmentation:
      return lambda imgs, imgs_na: (tf.ensure_shape(imgs, imgs_shape), {
          'images_without_photo_aug': tf.ensure_shape(imgs_na, imgs_shape)
      })
    else:
      return lambda imgs: (tf.ensure_shape(imgs, imgs_shape), {})

  train_ds = train_datasets[0]
  # Perform data augmentation
  # This cannot handle occlusions at the moment.
  if apply_augmentation:
    train_ds = train_ds.map(augmentation_fn)
  train_ds = train_ds.batch(batch_size)
  train_ds = train_ds.prefetch(1)
  train_ds = train_ds.map(_ensure_shapes())
  train_it = tf.compat.v1.data.make_one_shot_iterator(train_ds)
  return train_it


def make_eval_function(eval_on, height, width, progress_bar, plot_dir,
                       num_plots):
  """Build an evaluation function for uflow.

  Args:
    eval_on: string of the format 'format0:path0;format1:path1', e.g.
       'kitti:/usr/local/home/...'.
    height: int, the height to which the images should be resized for inference.
    width: int, the width to which the images should be resized for inference.
    progress_bar: boolean, flag to indicate whether the function should print a
      progress_bar during evaluaton.
    plot_dir: string, optional path to a directory in which plots are saved (if
      num_plots > 0).
    num_plots: int, maximum number of qualitative results to plot for the
      evaluation.
  Returns:
    A pair consisting of an evaluation function and a list of strings
      that holds the keys of the evaluation result.
  """
  eval_functions_and_datasets = []
  eval_keys = []
  # Split strings according to pattern "format0:path0;format1:path1".
  for format_and_path in eval_on.split(';'):
    data_format, path = format_and_path.split(':')

    # Add a dataset based on format and path.
    if 'kitti' in data_format:
      if 'benchmark' in data_format:
        dataset = kitti.make_dataset(path, mode='test')
        eval_fn = kitti.benchmark
      else:
        dataset = kitti.make_dataset(path, mode='eval')
        eval_fn = partial(kitti.evaluate, prefix=data_format)
        eval_keys += kitti.list_eval_keys(prefix=data_format)
    elif 'chairs' in data_format or 'custom' in data_format:
      dataset = flow_dataset.make_dataset(path, mode='eval')
      eval_fn = partial(
          flow_dataset.evaluate,
          prefix=data_format,
          max_num_evals=1000,  # We do this to avoid evaluating on 22k samples.
          has_occlusion=False)
      eval_keys += flow_dataset.list_eval_keys(prefix=data_format)
    elif 'sintel' in data_format:
      if 'benchmark' in data_format:
        # pylint:disable=g-long-lambda
        # pylint:disable=cell-var-from-loop
        eval_fn = lambda uflow: sintel.benchmark(inference_fn=uflow.infer,
                                                 height=height, width=width,
                                                 sintel_path=path,
                                                 plot_dir=plot_dir,
                                                 num_plots=num_plots)
        if len(eval_on.split(';')) != 1:
          raise ValueError('Sintel benchmark should be done in isolation.')
        return eval_fn, []
      dataset = sintel.make_dataset(path, mode='eval-occlusion')
      eval_fn = partial(sintel.evaluate, prefix=data_format)
      eval_keys += sintel.list_eval_keys(prefix=data_format)
    else:
      print('Unknown data format "{}"'.format(data_format))
      continue

    dataset = dataset.prefetch(1)
    eval_functions_and_datasets.append((eval_fn, dataset))

  # Make an eval function that aggregates all evaluations.
  def eval_function(uflow):
    result = dict()
    for eval_fn, ds in eval_functions_and_datasets:
      results = eval_fn(
          uflow.infer, ds, height,
          width, progress_bar, plot_dir, num_plots)
      for k, v in results.items():
        result[k] = v
    return result

  return eval_function, eval_keys
