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

"""Data loading and evaluation utilities shared across multiple datasets.

Some datasets are very similar, so to prevent code duplication, shared utilities
are put into this class.
"""
# pylint:disable=g-importing-member
from collections import defaultdict
import sys
import time

import numpy as np
import tensorflow as tf

from uflow import uflow_plotting
from uflow import uflow_utils


def parse_data(proto,
               include_flow,
               height=None,
               width=None,
               include_occlusion=False,
               include_invalid=False,
               resize_gt_flow=True,
               gt_flow_shape=None):
  """Parse a data proto with flow.

  Args:
    proto: path to data proto file
    include_flow: bool, whether or not to include flow in the output
    height: int or None height to resize image to
    width: int or None width to resize image to
    include_occlusion: bool, whether or not to also return occluded pixels (will
      throw error if occluded pixels are not present)
    include_invalid: bool, whether or not to also return invalid pixels (will
      throw error if invalid pixels are not present)
    resize_gt_flow: bool, wether or not to resize flow ground truth as the image
    gt_flow_shape: list, shape of the original ground truth flow (only required
      to set a fixed ground truth flow shape for tensorflow estimator in case of
      supervised training at full resolution resize_gt_flow=False)

  Returns:
    images, flow: A tuple of (image1, image2), flow
  """

  # Parse context and image sequence from protobuffer.
  context_features = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
  }
  sequence_features = {
      'images': tf.io.FixedLenSequenceFeature([], tf.string),
  }

  if include_invalid:
    sequence_features['invalid_masks'] = tf.io.FixedLenSequenceFeature(
        [], tf.string)

  if include_flow:
    context_features['flow_uv'] = tf.io.FixedLenFeature([], tf.string)

  if include_occlusion:
    context_features['occlusion_mask'] = tf.io.FixedLenFeature([], tf.string)

  context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
      proto,
      context_features=context_features,
      sequence_features=sequence_features,
  )

  def deserialize(s, dtype, dims):
    return tf.reshape(
        tf.io.decode_raw(s, dtype),
        [context_parsed['height'], context_parsed['width'], dims])

  images = tf.map_fn(
      lambda s: deserialize(s, tf.uint8, 3),
      sequence_parsed['images'],
      dtype=tf.uint8)

  images = tf.image.convert_image_dtype(images, tf.float32)
  if height is not None and width is not None:
    images = uflow_utils.resize(images, height, width, is_flow=False)
  output = [images]

  if include_flow:
    flow_uv = deserialize(context_parsed['flow_uv'], tf.float32, 2)
    flow_uv = flow_uv[Ellipsis, ::-1]
    if height is not None and width is not None and resize_gt_flow:
      flow_uv = uflow_utils.resize(flow_uv, height, width, is_flow=True)
    else:
      if gt_flow_shape is not None:
        flow_uv.set_shape(gt_flow_shape)
    # To be consistent with uflow internals, we flip the ordering of flow.
    output.append(flow_uv)
    # create valid mask
    flow_valid = tf.ones_like(flow_uv[Ellipsis, :1], dtype=tf.float32)
    output.append(flow_valid)

  if include_occlusion:
    occlusion_mask = deserialize(context_parsed['occlusion_mask'], tf.uint8, 1)
    if height is not None and width is not None:
      occlusion_mask = uflow_utils.resize(
          occlusion_mask, height, width, is_flow=False)
    output.append(occlusion_mask)

  if include_invalid:
    invalid_masks = tf.map_fn(
        lambda s: deserialize(s, tf.uint8, 1),
        sequence_parsed['invalid_masks'],
        dtype=tf.uint8)
    if height is not None and width is not None:
      invalid_masks = uflow_utils.resize(
          invalid_masks, height, width, is_flow=False)
    output.append(invalid_masks)

  # Only put the output in a list if there are more than one items in there.
  if len(output) == 1:
    output = output[0]

  return output


def compute_f_metrics(mask_prediction, mask_gt, num_thresholds=40):
  """Return a dictionary of the true positives, etc. for two binary masks."""
  results = defaultdict(dict)
  mask_prediction = tf.cast(mask_prediction, tf.float32)
  mask_gt = tf.cast(mask_gt, tf.float32)
  for threshold in np.linspace(0, 1, num_thresholds):
    mask_thresh = tf.cast(
        tf.math.greater(mask_prediction, threshold), tf.float32)
    true_pos = tf.cast(tf.math.count_nonzero(mask_thresh * mask_gt), tf.float32)
    true_neg = tf.math.count_nonzero((mask_thresh - 1) * (mask_gt - 1))
    false_pos = tf.cast(
        tf.math.count_nonzero(mask_thresh * (mask_gt - 1)), tf.float32)
    false_neg = tf.cast(
        tf.math.count_nonzero((mask_thresh - 1) * mask_gt), tf.float32)
    results[threshold]['tp'] = true_pos
    results[threshold]['fp'] = false_pos
    results[threshold]['fn'] = false_neg
    results[threshold]['tn'] = true_neg
  return results


def get_fmax_and_best_thresh(results):
  """Select which threshold produces the best f1 score."""
  fmax = -1.
  best_thresh = -1.
  for thresh, metrics in results.items():
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-6)
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    if f1 > fmax:
      fmax = f1
      best_thresh = thresh
  return fmax, best_thresh


def evaluate(
    inference_fn,
    dataset,
    height,
    width,
    progress_bar=False,
    plot_dir='',
    num_plots=0,
    max_num_evals=10000,
    prefix='',
    has_occlusion=True,
):
  """Evaluate an inference function for flow.

  Args:
    inference_fn: An inference function that produces a flow_field from two
      images, e.g. the infer method of UFlow.
    dataset: A dataset produced by the method above with for_eval=True.
    height: int, the height to which the images should be resized for inference.
    width: int, the width to which the images should be resized for inference.
    progress_bar: boolean, flag to indicate whether the function should print a
      progress_bar during evaluaton.
    plot_dir: string, optional path to a directory in which plots are saved (if
      num_plots > 0).
    num_plots: int, maximum number of qualitative results to plot for the
      evaluation.
    max_num_evals: int, maxmim number of evaluations.
    prefix: str, prefix to prepend to filenames for saved plots and for keys in
      results dictionary.
    has_occlusion: bool indicating whether or not the dataset includes ground
      truth occlusion.

  Returns:
    A dictionary of floats that represent different evaluation metrics. The keys
    of this dictionary are returned by the method list_eval_keys (see below).
  """

  eval_start_in_s = time.time()

  it = tf.compat.v1.data.make_one_shot_iterator(dataset)
  epe_occ = []  # End point errors.
  errors_occ = []
  inference_times = []
  all_occlusion_results = defaultdict(lambda: defaultdict(int))

  plot_count = 0
  eval_count = -1
  for test_batch in it:

    if eval_count >= max_num_evals:
      break

    eval_count += 1
    if eval_count >= max_num_evals:
      break

    if progress_bar:
      sys.stdout.write(':')
      sys.stdout.flush()

    if has_occlusion:
      (image_batch, flow_gt, _, occ_mask_gt) = test_batch
    else:
      (image_batch, flow_gt, _) = test_batch
      occ_mask_gt = tf.ones_like(flow_gt[Ellipsis, -1:])
    # pylint:disable=cell-var-from-loop
    # pylint:disable=g-long-lambda
    f = lambda: inference_fn(
        image_batch[0],
        image_batch[1],
        input_height=height,
        input_width=width,
        infer_occlusion=True)
    inference_time_in_ms, (flow, soft_occlusion_mask) = uflow_utils.time_it(
        f, execute_once_before=eval_count == 1)
    inference_times.append(inference_time_in_ms)

    if not has_occlusion:
      best_thresh = .5
    else:
      f_dict = compute_f_metrics(soft_occlusion_mask, occ_mask_gt)
      best_thresh = -1.
      best_f_score = -1.
      for thresh, metrics in f_dict.items():
        precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-6)
        recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        if f1 > best_f_score:
          best_thresh = thresh
          best_f_score = f1
        all_occlusion_results[thresh]['tp'] += metrics['tp']
        all_occlusion_results[thresh]['fp'] += metrics['fp']
        all_occlusion_results[thresh]['tn'] += metrics['tn']
        all_occlusion_results[thresh]['fn'] += metrics['fn']

    final_flow = flow
    endpoint_error_occ = tf.reduce_sum(
        input_tensor=(final_flow - flow_gt)**2, axis=-1, keepdims=True)**0.5
    gt_flow_abs = tf.reduce_sum(
        input_tensor=flow_gt**2, axis=-1, keepdims=True)**0.5
    outliers_occ = tf.cast(
        tf.logical_and(endpoint_error_occ > 3.,
                       endpoint_error_occ > 0.05 * gt_flow_abs), 'float32')
    epe_occ.append(tf.reduce_mean(input_tensor=endpoint_error_occ))
    errors_occ.append(tf.reduce_mean(input_tensor=outliers_occ))

    if plot_dir and plot_count < num_plots:
      plot_count += 1
      mask_thresh = tf.cast(
          tf.math.greater(soft_occlusion_mask, best_thresh), tf.float32)
      uflow_plotting.complete_paper_plot(
          plot_dir,
          plot_count,
          image_batch[0].numpy(),
          image_batch[1].numpy(),
          final_flow.numpy(),
          flow_gt.numpy(),
          np.ones_like(mask_thresh.numpy()),
          1. - mask_thresh.numpy(),
          1. - occ_mask_gt.numpy().astype('float32'),
          frame_skip=None)
  if progress_bar:
    sys.stdout.write('\n')
    sys.stdout.flush()

  fmax, best_thresh = get_fmax_and_best_thresh(all_occlusion_results)
  eval_stop_in_s = time.time()

  results = {
      'occl-f-max': fmax,
      'best-occl-thresh': best_thresh,
      'EPE': np.mean(np.array(epe_occ)),
      'ER': np.mean(np.array(errors_occ)),
      'inf-time(ms)': np.mean(inference_times),
      'eval-time(s)': eval_stop_in_s - eval_start_in_s
  }
  if prefix:
    return {prefix + '-' + k: v for k, v in results.items()}
  return results


def list_eval_keys(prefix=''):
  """List the keys of the dictionary returned by the evaluate function."""
  keys = [
      'EPE', 'ER', 'inf-time(ms)', 'eval-time(s)', 'occl-f-max',
      'best-occl-thresh'
  ]
  if prefix:
    return [prefix + '-' + k for k in keys]
  return keys
