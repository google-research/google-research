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

"""Utilities for data loading that are shared across multiple datasets.

Some datasets are very similar, so to prevent code duplication, shared utilities
are put into this class.
"""

# pylint:skip-file
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

from smurf import smurf_utils
from smurf import smurf_plotting


def parse_data(proto,
               include_flow,
               height=None,
               width=None,
               include_occlusion=False,
               include_invalid=False,
               resize_gt_flow=True,
               include_image_path=False,
               gt_flow_shape=None,
               include_segments=False):
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
    include_image_path: bool, if True, return the string for the key
      "image1_path" alongside the data.
    gt_flow_shape: list, shape of the original ground truth flow (only required
      to set a fixed ground truth flow shape for tensorflow estimator in case of
      supervised training at full resolution resize_gt_flow=False)
    include_segments: bool, if True, include the Sintel segmentation data.

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

  if include_segments:
    sequence_features['segments'] = tf.io.FixedLenSequenceFeature(
        [], tf.string)
    sequence_features['segments_invalid'] = tf.io.FixedLenSequenceFeature(
        [], tf.string)

  if include_image_path:
    context_features['image1_path'] = tf.io.FixedLenFeature((), tf.string)

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
    images = smurf_utils.resize(images, height, width, is_flow=False)
  output = {'images': images}

  if include_flow:
    flow_uv = deserialize(context_parsed['flow_uv'], tf.float32, 2)
    flow_uv = flow_uv[Ellipsis, ::-1]
    # Flying things has some images with erroneously large flow.
    # Mask out any values above / below 1000.
    invalid_cond = tf.math.logical_or(tf.greater(flow_uv, 1000),
                                      tf.less(flow_uv, -1000))
    mask = tf.where(invalid_cond, tf.zeros_like(flow_uv), tf.ones_like(flow_uv))
    flow_valid = tf.reduce_min(mask, axis=-1, keepdims=True)
    if height is not None and width is not None and resize_gt_flow:
      flow_uv = smurf_utils.resize(flow_uv, height, width, is_flow=True)
      flow_valid = smurf_utils.resize(flow_valid, height, width, is_flow=False)
    else:
      if gt_flow_shape is not None:
        flow_uv.set_shape(gt_flow_shape)
        flow_valid.set_shape((gt_flow_shape[0], gt_flow_shape[1], 1))
    # To be consistent with SMURF internals, we flip the ordering of flow.
    # create valid mask
    flow_valid = tf.ones_like(flow_uv[Ellipsis, :1], dtype=tf.float32)
    output['flow_valid'] = flow_valid
    output['flow'] = flow_uv

  if include_occlusion:
    occlusion_mask = deserialize(context_parsed['occlusion_mask'], tf.uint8, 1)
    if height is not None and width is not None:
      occlusion_mask = smurf_utils.resize(
          occlusion_mask, height, width, is_flow=False)
    output['occlusions'] = occlusion_mask

  if include_invalid:
    invalid_masks = tf.map_fn(
        lambda s: deserialize(s, tf.uint8, 1),
        sequence_parsed['invalid_masks'],
        dtype=tf.uint8)
    if height is not None and width is not None:
      invalid_masks = smurf_utils.resize(
          invalid_masks, height, width, is_flow=False)
    output['flow_valid'] = 1. - invalid_masks

  if include_image_path:
    output['image1_path'] = context_parsed['image1_path']

  if include_segments:
    segments = tf.map_fn(
        lambda s: deserialize(s, tf.uint8, 3),
        sequence_parsed['segments'],
        dtype=tf.uint8)
    segments = tf.image.convert_image_dtype(segments, tf.float32)
    segments_invalid = tf.map_fn(
        lambda s: deserialize(s, tf.uint8, 1),
        sequence_parsed['segments_invalid'],
        dtype=tf.uint8)
    segments_invalid = tf.image.convert_image_dtype(segments_invalid,
                                                    tf.float32)
    segments = tf.image.resize(segments, (height, width), method='nearest')
    segments_invalid = tf.image.resize(
        segments_invalid, (height, width), method='nearest')
    output['segments'] = segments
    output['segments_invalid'] = segments_invalid

  return output


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
    weights=None,
):
  """Evaluate an inference function for flow.

  Args:
    inference_fn: An inference function that produces a flow_field from two
      images, e.g. the infer method of SMURF.
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
    weights: unsupervised loss weights

  Returns:
    A dictionary of floats that represent different evaluation metrics. The keys
    of this dictionary are returned by the method list_eval_keys (see below).
  """

  eval_start_in_s = time.time()

  it = tf.compat.v1.data.make_one_shot_iterator(dataset)
  epe_occ = []  # End point errors.
  errors_occ = []
  inference_times = []
  unsuper_losses = []
  all_occlusion_results = defaultdict(lambda: defaultdict(int))

  plot_count = 0
  eval_count = -1
  for test_batch in it:

    image_batch = test_batch['images']
    flow_gt = test_batch['flow']
    flow_valid = test_batch['flow_valid']
    if has_occlusion:
      occ_mask_gt = test_batch['occlusions']
    else:
      occ_mask_gt = tf.ones_like(flow_valid)

    if eval_count >= max_num_evals:
      break

    eval_count += 1
    if eval_count >= max_num_evals:
      break

    if progress_bar:
      sys.stdout.write(':')
      sys.stdout.flush()

    f = lambda: inference_fn(
        image_batch[0],
        image_batch[1],
        input_height=height,
        input_width=width,
        infer_occlusion=True,
        infer_bw=True)

    inference_time_in_ms, (flow, soft_occlusion_mask,
                           flow_bw) = smurf_utils.time_it(
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
    endpoint_error_occ = epe_elementwise(final_flow, flow_gt) * flow_valid
    outliers_occ = outliers_elementwise(final_flow, flow_gt) * flow_valid
    # NOTE(austinstone): The unsupervised loss function expects occluded areas
    # to be zeros, whereas the occlusion mask above returns 1s in areas
    # of occlusions. The unsupervised loss function below assumes some
    # parameters (smoothness edge info) is left at the default values.
    if weights is not None:
      inv_mask = tf.expand_dims(1. - soft_occlusion_mask, axis=0)
      occlusion_estimation_fn = lambda forward_flow, backward_flow: inv_mask
      loss_dict = smurf_utils.unsupervised_loss(
          images=tf.expand_dims(image_batch, axis=0),
          flows={
              (0, 1, 'augmented-student'): [tf.expand_dims(flow, axis=0)],
              (1, 0, 'augmented-student'): [tf.expand_dims(flow_bw, axis=0)]
          },
          weights=weights,
          occlusion_estimation_fn=occlusion_estimation_fn)
      loss_dict['total_loss'] = sum(loss_dict.values())
    else:
      loss_dict = {}
    unsuper_losses.append(loss_dict)
    epe_occ.append(tf.reduce_mean(input_tensor=endpoint_error_occ))
    errors_occ.append(tf.reduce_mean(input_tensor=outliers_occ))

    if plot_dir and plot_count < num_plots:
      plot_count += 1
      mask_thresh = tf.cast(
          tf.math.greater(soft_occlusion_mask, best_thresh), tf.float32)
      smurf_plotting.complete_paper_plot(
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
      'eval-time(s)': eval_stop_in_s - eval_start_in_s,
  }
  for k in unsuper_losses[0].keys():
    results.update({k: np.mean([l[k] for l in unsuper_losses])})

  if prefix:
    return {prefix + '-' + k: v for k, v in results.items()}
  return results


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


def list_eval_keys(prefix=''):
  """List the keys of the dictionary returned by the evaluate function."""
  keys = [
      'EPE', 'ER', 'inf-time(ms)', 'eval-time(s)', 'occl-f-max',
      'best-occl-thresh'
  ]
  if prefix:
    return [prefix + '-' + k for k in keys]
  return keys


def epe_elementwise(estimation, groundtruth):
  """Computes the endpoint-error of each flow vector."""
  return tf.reduce_sum(
      (estimation - groundtruth)**2, axis=-1, keepdims=True)**0.5


def length_elementwise(flow):
  """Computes the length of each flow vector."""
  return tf.reduce_sum(flow**2, axis=-1, keepdims=True)**0.5


def outliers_elementwise(estimation, groundtruth, epe_threshold=3.0):
  """Computes the outlier criteria for the error rate per flow vector."""
  epe = epe_elementwise(estimation, groundtruth)
  length = length_elementwise(groundtruth)
  return tf.cast(
      tf.logical_and(epe > epe_threshold, epe > 0.05 * length), 'float32')


def angular_error_elementwise(estimation, groundtruth):
  """Computes the anuglar-error of each flow vector."""
  h, w, _ = tf.unstack(tf.shape(estimation))
  time_dim = tf.ones([h, w, 1])

  e_t = tf.concat([estimation, time_dim], -1)
  g_t = tf.concat([groundtruth, time_dim], -1)
  e_t = e_t / tf.reduce_sum(e_t**2, axis=-1, keepdims=True)**0.5
  g_t = g_t / tf.reduce_sum(g_t**2, axis=-1, keepdims=True)**0.5
  return tf.math.acos(
      tf.clip_by_value(tf.reduce_sum(e_t * g_t, -1, keepdims=True), -1., 1.))
