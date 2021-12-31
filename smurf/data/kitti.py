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

"""Dataset generation (for train. and eval.) and evaluation method for KITTI."""

# pylint: skip-file
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # None-interactive plots do not need tk
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow as tf

from smurf import smurf_plotting
from smurf import smurf_utils
from smurf.data import data_utils


def parse_data(proto, height, width):
  """Parse features from byte-encoding to the correct type and shape.

  Args:
    proto: Encoded data in proto / tf-sequence-example format.
    height: int, desired image height.
    width: int, desired image width.

  Returns:
    A sequence of images as tf.Tensor of shape
    [sequence length, height, width, 3].
  """

  # Parse context and image sequence from protobuffer.
  unused_context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
      proto,
      context_features={
          'height': tf.io.FixedLenFeature([], tf.int64),
          'width': tf.io.FixedLenFeature([], tf.int64)
      },
      sequence_features={
          'images': tf.io.FixedLenSequenceFeature([], tf.string)
      })

  # Deserialize images to float32 tensors.
  def deserialize(image_raw):
    image_uint = tf.image.decode_png(image_raw)
    image_float = tf.image.convert_image_dtype(image_uint, tf.float32)
    return image_float
  images = tf.map_fn(deserialize, sequence_parsed['images'], dtype=tf.float32)

  # Resize images.
  if height is not None and width is not None:
    images = smurf_utils.resize(images, height, width, is_flow=False)

  return images


def parse_supervised_train_data(proto, height, width, resize_gt_flow):
  """Parse proto from byte-encoding to the correct type and shape.

  Args:
    proto: Encoded data in proto / tf-sequence-example format.
    height: int, desired image height.
    width: int, desired image width.
    resize_gt_flow: bool, wether or not to resize flow according to the images

  Returns:
    A tuple of tf.Tensors for images, flow_uv, flow_valid, where uv represents
    the flow field and valid a mask for which entries are valid (this uses the
    occ version that includes all flow vectors). The images and the
    corresponding flow field are resized to the specified [height, width].
  """
  # Reuse the evaluation parser to parse the supervised data.
  data_dict = parse_eval_data(proto)
  images = data_dict['images']
  flow_uv_occ = data_dict['flow_uv_occ']
  flow_valid_occ = data_dict['flow_valid_occ']
  flow_valid_occ = tf.cast(flow_valid_occ, tf.float32)

  if not resize_gt_flow or height is None or width is None:
    # Crop to a size that fits all KITTI 2015 image resolutions. Because the
    # first 156 sequences have a resolution of 375x1242,the remaining 44
    # sequences include resolutions of 370x1224, 374x1238, and 376x1241.
    _, orig_height, orig_width, _ = tf.unstack(tf.shape(images))
    offset_height = tf.cast((orig_height - 370) / 2, tf.int32)
    offset_width = tf.cast((orig_width - 1224) / 2, tf.int32)
    images = tf.image.crop_to_bounding_box(
        images, offset_height=offset_height, offset_width=offset_width,
        target_height=370, target_width=1224)
    flow_uv_occ = tf.image.crop_to_bounding_box(
        flow_uv_occ, offset_height=offset_height, offset_width=offset_width,
        target_height=370, target_width=1224)
    flow_valid_occ = tf.image.crop_to_bounding_box(
        flow_valid_occ, offset_height=offset_height, offset_width=offset_width,
        target_height=370, target_width=1224)

  # resize images
  if height is not None and width is not None:
    images = smurf_utils.resize(images, height, width, is_flow=False)

  if resize_gt_flow and height is not None and width is not None:
    # resize flow and swap label order
    flow_uv, flow_valid = smurf_utils.resize(
        flow_uv_occ[Ellipsis, ::-1],
        height,
        width,
        is_flow=True,
        mask=flow_valid_occ)
  else:
    # only swap label order
    flow_uv = flow_uv_occ[Ellipsis, ::-1]
    flow_valid = flow_valid_occ
    # set shape to work with tf estimator
    flow_uv.set_shape([370, 1224, 2])
    flow_valid.set_shape([370, 1224, 1])

  return {'images': images, 'flow': flow_uv, 'flow_valid': flow_valid}


def parse_eval_data(proto):
  """Parse eval proto from byte-encoding to the correct type and shape.

  Args:
    proto: Encoded data in proto / tf-sequence-example format containing context
      features height, width, flow_uv_occ, flow_uv_noc, flow_valid_occ,
      flow_valid_noc and sequence features images, as generated by
      convert_KITTI_flow_to_tfrecords.py

  Returns:
    A tuple of tf.Tensors for images, flow_uv_occ, flow_uv_noc, flow_valid_occ,
    flow_valid_noc, where uv represents the flow field and valid a mask for
    which entries are valid, occ includes all flow vectors and noc excludes
    those that are not visible in the next frame.
  """

  # Parse context and image sequence from protobuffer.
  context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
      proto,
      context_features={
          'height': tf.io.FixedLenFeature([], tf.int64),
          'width': tf.io.FixedLenFeature([], tf.int64),
          'flow_uv_occ': tf.io.FixedLenFeature([], tf.string),
          'flow_uv_noc': tf.io.FixedLenFeature([], tf.string),
          'flow_valid_occ': tf.io.FixedLenFeature([], tf.string),
          'flow_valid_noc': tf.io.FixedLenFeature([], tf.string),
      },
      sequence_features={
          'images': tf.io.FixedLenSequenceFeature([], tf.string)
      })

  def deserialize(s, dtype, dims):
    return tf.reshape(
        tf.io.decode_raw(s, dtype),
        [context_parsed['height'], context_parsed['width'], dims])

  images = tf.map_fn(
      lambda s: deserialize(s, tf.uint8, 3),
      sequence_parsed['images'],
      dtype=tf.uint8)
  images = tf.image.convert_image_dtype(images, tf.float32)
  flow_uv_occ = deserialize(context_parsed['flow_uv_occ'], tf.float32, 2)
  flow_uv_noc = deserialize(context_parsed['flow_uv_noc'], tf.float32, 2)
  flow_valid_occ = deserialize(context_parsed['flow_valid_occ'], tf.uint8, 1)
  flow_valid_noc = deserialize(context_parsed['flow_valid_noc'], tf.uint8, 1)
  return {'images': images, 'flow_uv_occ': flow_uv_occ,
          'flow_uv_noc': flow_uv_noc, 'flow_valid_occ': flow_valid_occ,
          'flow_valid_noc': flow_valid_noc}


def parse_test_data(proto):
  """Parse eval proto from byte-encoding to the correct type and shape.

  Args:
    proto: Encoded data in proto / tf-sequence-example format containing context
      features height, width, flow_uv_occ, flow_uv_noc, flow_valid_occ,
      flow_valid_noc and sequence features images, as generated by
      convert_KITTI_flow_to_tfrecords.py

  Returns:
    A tuple of tf.Tensors for images, flow_uv_occ, flow_uv_noc, flow_valid_occ,
    flow_valid_noc, where uv represents the flow field and valid a mask for
    which entries are valid, occ includes all flow vectors and noc excludes
    those that are not visible in the next frame.
  """

  # Parse context and image sequence from protobuffer.
  context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
      proto,
      context_features={
          'height': tf.io.FixedLenFeature([], tf.int64),
          'width': tf.io.FixedLenFeature([], tf.int64),
      },
      sequence_features={
          'images': tf.io.FixedLenSequenceFeature([], tf.string)
      })

  def deserialize(s, dtype, dims):
    return tf.reshape(
        tf.io.decode_raw(s, dtype),
        [context_parsed['height'], context_parsed['width'], dims])

  images = tf.map_fn(
      lambda s: deserialize(s, tf.uint8, 3),
      sequence_parsed['images'],
      dtype=tf.uint8)
  images = tf.image.convert_image_dtype(images, tf.float32)
  return {'images': images}


def make_dataset(path,
                 mode,
                 seq_len=2,
                 shuffle_buffer_size=0,
                 height=None,
                 width=None,
                 resize_gt_flow=True,
                 seed=41):
  """Make a dataset for training or evaluating SMURF.

  Args:
    path: string, in the format of 'some/path/dir1,dir2,dir3' to load all files
      in some/path/dir1, some/path/dir2, and some/path/dir3.
    mode: string, one of ['train', 'eval', 'test'] to switch between loading
    seq_len: The number of images in the sequence to return (if applicable).
    training data, evaluation data, and test data, which have different formats.
    shuffle_buffer_size: int, size of the shuffle buffer; no shuffling if 0.
    height: int, height for reshaping the images (only if for_eval=False)
      because reshaping for eval is more complicated and done in the evaluate
      function through smurf.inference.
    width: int, width for reshaping the images (only if for_eval=False).
    resize_gt_flow: bool, indicates if ground truth flow should be resized
      during traing or not (only relevant for supervised training)
    seed: int, controls the shuffling of the data shards.

  Returns:
    A tf.dataset of image sequences for training and of a tuple of things for
    evaluation (see parse functions above). The dataset still requires batching
    and prefetching before using it to make an iterator.
  """

  if ',' in path:
    l = path.split(',')
    d = '/'.join(l[0].split('/')[:-1])
    l[0] = l[0].split('/')[-1]
    paths = [os.path.join(d, x) for x in l]
  else:
    paths = [path]

  # Generate list of filenames.
  files = [
      os.path.join(d, f)
      for d in paths
      for f in tf.io.gfile.listdir(d)
  ]
  num_files = len(files)
  if 'train' in mode:
    rgen = np.random.RandomState(seed)
    rgen.shuffle(files)
  ds = tf.data.Dataset.from_tensor_slices(files)

  if mode == 'eval':
    if height is not None or width is not None:
      raise ValueError('for_eval is incompatible with height/width')
    if shuffle_buffer_size:
      raise ValueError('for_eval is incompatible with shuffle_buffer_size')
    if seq_len != 2:
      raise ValueError('for_eval only compatible with seq_len == 2.')
    ds = ds.map(tf.data.TFRecordDataset)
    # Parse each element of the subsequences and unbatch the result.
    parse_fn = parse_eval_data

    ds = ds.interleave(
        lambda x: x.map(
            lambda y: parse_fn(y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE),
        cycle_length=min(10, num_files),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif mode == 'test':
    if height is not None or width is not None:
      raise ValueError('for_eval is incompatible with height/width')
    if shuffle_buffer_size:
      raise ValueError('for_eval is incompatible with shuffle_buffer_size')
    if seq_len != 2:
      raise ValueError('for_eval only compatible with seq_len == 2.')
    ds = ds.map(tf.data.TFRecordDataset)
    # Parse each element of the subsequences and unbatch the result.
    ds = ds.flat_map(lambda x: x.map(
        lambda y: parse_test_data(y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE))
  elif mode == 'train' or mode == 'video':
    if shuffle_buffer_size:
      ds = ds.shuffle(num_files)
    # Create a nested dataset.
    ds = ds.map(tf.data.TFRecordDataset)
    # Parse each element of the subsequences and unbatch the result.
    ds = ds.map(lambda x: x.map(
        lambda y: parse_data(y, height, width),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch())
    # Slide a window over each dataset, combine either by interleaving or by
    # sequencing the result (produces a a nested dataset)
    window_fn = lambda x: x.window(size=seq_len, shift=1, drop_remainder=True)
    # Interleave subsequences (too long cycle length causes memory issues).
    ds = ds.interleave(
        window_fn,
        cycle_length=1 if 'video' in mode else min(10, num_files),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle_buffer_size:
      # Shuffle subsequences.
      ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Put repeat after shuffle for better data mixing.
    ds = ds.repeat()
    # Flatten the nested dataset into a batched dataset.
    ds = ds.flat_map(lambda x: x.batch(seq_len))
    ds = ds.map(lambda x: {'images': x})
    # Prefetch a number of batches because reading new ones can take much longer
    # when they are from new files.
    ds = ds.prefetch(10)

  elif 'train' in mode and 'sup' in mode:
    if shuffle_buffer_size:
      ds = ds.shuffle(num_files)
    # Create a nested dataset.
    ds = ds.map(tf.data.TFRecordDataset)
    # Parse each element of the subsequences and unbatch the result.
    ds = ds.interleave(lambda x: x.map(
        lambda y: parse_supervised_train_data(y, height, width, resize_gt_flow),
        num_parallel_calls=tf.data.experimental.AUTOTUNE),
    cycle_length=min(10, num_files),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle_buffer_size:
      # Shuffle subsequences.
      ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Put repeat after shuffle for better data mixing.
    ds = ds.repeat()
    # Prefetch a number of batches because reading new ones can take much longer
    # when they are from new files.
    ds = ds.prefetch(10)
  else:
    raise NotImplementedError('Unknown mode.')

  return ds


def evaluate(inference_fn,
             dataset,
             height,
             width,
             progress_bar=False,
             plot_dir='',
             num_plots=0,
             prefix='kitti'):
  """Evaluate an iference function for flow with a kitti eval dataset.

  Args:
    inference_fn: An inference function that produces a flow_field from two
      images, e.g. the infer method of SMURF.
    dataset: A dataset produced by the method above with for_eval=True.
    height: int, the height to which the images should be resized for inference.
    width: int, the width to which the images should be resized for inference.
    progress_bar: boolean, flag to indicate whether the function should print a
      progress_bar during evaluaton.
    plot_dir: string, optional path to a directory in which plots are saved
      (if num_plots > 0).
    num_plots: int, maximum number of qualitative results to plot for the
      evaluation.

  Returns:
    A dictionary of floats that represent different evaluation metrics. The keys
    of this dictionary are returned by the method list_eval_keys (see below).
  """

  eval_start_in_s = time.time()

  it = tf.compat.v1.data.make_one_shot_iterator(dataset)
  epe_occ = []  # End point errors.
  errors_occ = []
  valid_occ = []
  epe_noc = []  # End point errors.
  errors_noc = []
  valid_noc = []
  inference_times = []
  all_occlusion_results = defaultdict(lambda: defaultdict(int))

  for i, test_batch in enumerate(it):

    if progress_bar:
      sys.stdout.write(':')
      sys.stdout.flush()

    image_batch = test_batch['images']
    flow_uv_occ = test_batch['flow_uv_occ']
    flow_uv_noc = test_batch['flow_uv_noc']
    flow_valid_occ = test_batch['flow_valid_occ']
    flow_valid_noc = test_batch['flow_valid_noc']

    flow_valid_occ = tf.cast(flow_valid_occ, 'float32')
    flow_valid_noc = tf.cast(flow_valid_noc, 'float32')

    f = lambda: inference_fn(
        image_batch[0],
        image_batch[1],
        input_height=height,
        input_width=width,
        infer_occlusion=True)
    inference_time_in_ms, (flow, soft_occlusion_mask) = smurf_utils.time_it(
        f, execute_once_before=i == 0)
    inference_times.append(inference_time_in_ms)

    occ_mask_gt = flow_valid_occ - flow_valid_noc
    f_dict = data_utils.compute_f_metrics(soft_occlusion_mask * flow_valid_occ,
                                          occ_mask_gt * flow_valid_occ)
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

    mask_thresh = tf.cast(
        tf.math.greater(soft_occlusion_mask, best_thresh), tf.float32)
    # Image coordinates are swapped in labels
    final_flow = flow[Ellipsis, ::-1]

    endpoint_error_occ = data_utils.epe_elementwise(final_flow, flow_uv_occ)
    outliers_occ = data_utils.outliers_elementwise(final_flow, flow_uv_occ)

    endpoint_error_noc = data_utils.epe_elementwise(final_flow, flow_uv_noc)
    outliers_noc = data_utils.outliers_elementwise(final_flow, flow_uv_noc)

    epe_occ.append(tf.reduce_sum(input_tensor=flow_valid_occ * endpoint_error_occ))
    errors_occ.append(tf.reduce_sum(input_tensor=flow_valid_occ * outliers_occ))
    valid_occ.append(tf.reduce_sum(input_tensor=flow_valid_occ))

    epe_noc.append(tf.reduce_sum(input_tensor=flow_valid_noc * endpoint_error_noc))
    errors_noc.append(tf.reduce_sum(input_tensor=flow_valid_noc * outliers_noc))
    valid_noc.append(tf.reduce_sum(input_tensor=flow_valid_noc))

    if plot_dir and i < num_plots:
      smurf_plotting.complete_paper_plot(plot_dir, i,
                                         image_batch[0].numpy(),
                                         image_batch[1].numpy(),
                                         final_flow.numpy(), flow_uv_occ.numpy(),
                                         flow_valid_occ.numpy(),
                                         (1. - mask_thresh).numpy(),
                                         (1. - occ_mask_gt).numpy(),
                                         frame_skip=None)
  if progress_bar:
    sys.stdout.write('\n')
    sys.stdout.flush()

  fmax, best_thresh = data_utils.get_fmax_and_best_thresh(all_occlusion_results)
  eval_stop_in_s = time.time()

  results = {
      prefix + '-occl-f-max': fmax,
      prefix + '-best-occl-thresh': best_thresh,
      prefix + '-EPE(occ)':
          np.clip(np.mean(np.array(epe_occ) / np.array(valid_occ)), 0.0, 50.0),
      prefix + '-ER(occ)':
          np.mean(np.array(errors_occ) / np.array(valid_occ)),
      prefix + '-EPE(noc)':
          np.clip(np.mean(np.array(epe_noc) / np.array(valid_noc)), 0.0, 50.0),
      prefix + '-ER(noc)':
          np.mean(np.array(errors_noc) / np.array(valid_noc)),
      prefix + '-inf-time(ms)':
          np.mean(inference_times),
      prefix + '-eval-time(s)':
          eval_stop_in_s - eval_start_in_s,
  }
  return results


def list_eval_keys(prefix='kitti'):
  """List the keys of the dictionary returned by the evaluate function."""
  return [
      prefix + '-EPE(occ)', prefix + '-EPE(noc)', prefix + '-ER(occ)',
      prefix + '-ER(noc)', prefix + '-inf-time(ms)', prefix + '-eval-time(s)',
      prefix + '-occl-f-max', prefix + '-best-occl-thresh',
  ]


def benchmark(inference_fn, dataset, height, width, progress_bar=False,
              plot_dir='', unused_num_plots=0):
  """Evaluate an iference function for flow with a kitti eval dataset.

  Args:
    inference_fn: An inference function that produces a flow_field from two
      images, e.g. the infer method of SMURF.
    dataset: A dataset produced by the method above with for_eval=True.
    height: int, the height to which the images should be resized for inference.
    width: int, the width to which the images should be resized for inference.
    progress_bar: boolean, flag to indicate whether the function should print a
      progress_bar during evaluaton.
    plot_dir: string, optional path to a directory in which plots are saved
      (if num_plots > 0).
    num_plots: int, maximum number of qualitative results to plot for the
      evaluation.

  Returns:
    A dictionary of floats that represent different evaluation metrics. The keys
    of this dictionary are returned by the method list_eval_keys (see below).
  """

  it = tf.compat.v1.data.make_one_shot_iterator(dataset)
  inference_times = []

  for i, test_batch in enumerate(it):

    if progress_bar:
      sys.stdout.write(':')
      sys.stdout.flush()

    image_batch = test_batch['images']

    start_in_ms = time.time() * 1000
    flow = inference_fn(image_batch[0], image_batch[1],
                        input_height=height, input_width=width)
    stop_in_ms = time.time() * 1000
    inference_time_in_ms = stop_in_ms - start_in_ms
    inference_times.append(inference_time_in_ms)

    # Image coordinates are swapped in labels.
    final_flow = flow[Ellipsis, ::-1]
    flow_uint16 = tf.cast(final_flow * 64.0 + 2 ** 15, tf.uint16)
    flow_uint16 = tf.concat([flow_uint16, tf.ones_like(flow_uint16[Ellipsis, :1],
                                                       dtype=tf.uint16)], axis=-1)
    flow_png = tf.image.encode_png(flow_uint16)
    filename = os.path.join(plot_dir, '{:06d}_10.png'.format(i))
    tf.io.write_file(filename, flow_png)
    # with open(, 'wb+') as f:
    #   f.write(flow_png)

  print('Average inference time: {}ms'.format(np.mean(inference_time_in_ms)))
