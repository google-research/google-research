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

"""Data loader for sintel."""

import glob
import os
# pylint:disable=g-bad-import-order
import matplotlib
# pylint:disable=g-import-not-at-top
matplotlib.use('Agg')  # None-interactive plots do not need tk
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import tensorflow as tf

from smurf import smurf_plotting
from smurf.data import data_utils
# pylint:disable=unused-import
from smurf.data.data_utils import evaluate
from smurf.data.data_utils import list_eval_keys
from smurf.data_conversion_scripts import conversion_utils


def make_dataset(path,
                 mode,
                 seq_len=2,
                 shuffle_buffer_size=0,
                 shuffle_files=None,
                 height=None,
                 width=None,
                 resize_gt_flow=True,
                 include_segments=False,
                 repeat=None,
                 seed=41):
  """Make a dataset for training or evaluating SMURF.

  Args:
    path: string, in the format of 'some/path/dir1,dir2,dir3' to load all files
      in some/path/dir1, some/path/dir2, and some/path/dir3.
    mode: string, one of ['train-clean', 'eval-clean', 'train-final',
      'eval-final] to switch between loading train / eval data from the clean or
      final renderings.
    seq_len: int length of sequence to return. Currently only 2 is supported.
    shuffle_buffer_size: int, size of the shuffle buffer; no shuffling if 0.
    shuffle_files: bool, whether to shuffle shard files list. Defaults to
      is ('train' in mode).
    height: int, height for reshaping the images (only if mode==train)
    width: int, width for reshaping the images (only if mode==train)
    resize_gt_flow: bool, indicates if ground truth flow should be resized
      during traing or not (only relevant for supervised training)
    include_segments: bool, indicates whether to include the sintel segmentation
      data.
    repeat: bool, whether to repeat the dataset.
    seed: int, controls the shuffling of the data shards.

  Returns:
    A tf.dataset of image sequences and ground truth flow for training
    (see parse functions above). The dataset still requires batching
    and prefetching before using it to make an iterator.
  """
  assert seq_len == 2
  if repeat is None:
    repeat = ('train' in mode)
  if shuffle_files is None:
    shuffle_files = ('train' in mode)

  if ',' in path:
    paths = []
    l = path.split(',')
    paths.append(l[0])
    for subpath in l[1:]:
      subpath_length = len(subpath.split('/'))
      basedir = '/'.join(l[0].split('/')[:-subpath_length])
      paths.append(os.path.join(basedir, subpath))
  else:
    paths = [path]

  # Generate list of filenames.
  # pylint:disable=g-complex-comprehension
  files = [
      os.path.join(d, f)
      for d in paths
      for f in tf.io.gfile.listdir(d)
  ]
  if shuffle_files:
    rgen = np.random.RandomState(seed=seed)
    rgen.shuffle(files)
  num_files = len(files)

  ds = tf.data.Dataset.from_tensor_slices(files)
  # Create a nested dataset.
  ds = ds.map(tf.data.TFRecordDataset)
  # Parse each element of the subsequences and unbatch the result
  # Do interleave rather than flat_map because it is much faster.
  include_flow = 'eval' in mode or 'sup' in mode
  include_occlusion = 'occlusion' in mode
  include_invalid = 'invalid' in mode
  # pylint:disable=g-long-lambda
  ds = ds.interleave(
      lambda x: x.map(
          lambda y: data_utils.parse_data(
              y, include_flow=include_flow, height=height, width=width,
              include_occlusion=include_occlusion,
              include_invalid=include_invalid,
              resize_gt_flow=resize_gt_flow,
              include_image_path='manual-split' in mode,
              gt_flow_shape=[436, 1024, 2],
              include_segments=include_segments),
          num_parallel_calls=tf.data.experimental.AUTOTUNE),
      cycle_length=1 if 'video' in mode else min(10, num_files),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if shuffle_buffer_size:
    # Shuffle image pairs.
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # Put repeat after shuffle for better data mixing.
  if 'train' in mode:
    ds = ds.repeat()
  # Prefetch a number of batches because reading new ones can take much longer
  # when they are from new files.
  ds = ds.prefetch(10)
  return ds


def benchmark_iterator(data_dir, output_dir):
  """Iterates through Sintel test data filepaths for benchmarking."""
  for data_type in ['clean', 'final']:
    output_folder = os.path.join(output_dir, data_type)
    if not os.path.exists(output_folder):
      os.mkdir(output_folder)

    input_folder = os.path.join(data_dir, 'test', data_type)

    # Directory with images.
    image_folders = sorted(glob.glob(input_folder + '/*'))

    if not image_folders:
      raise ValueError('Must pass path to raw MPI-Sintel-complete dataset. '
                       'Got instead: {}'.format(data_dir))

    def sort_by_frame_index(x):
      return int(os.path.basename(x).split('_')[1].split('.')[0])

    for image_folder in image_folders:
      images = glob.glob(image_folder + '/*png')

      images = sorted(images, key=sort_by_frame_index)
      image_pairs = zip(images[:-1], images[1:])

      for images in image_pairs:
        img1_path, img2_path = images
        tf.compat.v1.logging.info('im1 path: %s, im2 path: %s', img1_path,
                                  img2_path)
        image1_data = scipy.ndimage.imread(img1_path)
        image2_data = scipy.ndimage.imread(img2_path)
        folder_name = os.path.basename(os.path.dirname(img1_path))
        flow_output_path = os.path.join(output_folder, folder_name)
        if not os.path.exists(flow_output_path):
          os.mkdir(flow_output_path)
        frame_num = os.path.basename(img1_path).replace('_', '').replace('png',
                                                                         'flo')
        flow_output_path = os.path.join(flow_output_path, frame_num)
        yield image1_data, image2_data, flow_output_path


def benchmark(inference_fn,
              height,
              width,
              sintel_path,
              plot_dir='',
              num_plots=100):
  """Produce benchmark data."""

  assert plot_dir
  output_path = os.path.join(plot_dir, 'sintel-upload-ready')
  if not os.path.exists(output_path):
    os.mkdir(output_path)

  it = benchmark_iterator(sintel_path, output_path)

  plot_count = 0

  for index, test_batch in enumerate(it):

    tf.compat.v1.logging.info('Writing results for image number %d...', index)

    (image1, image2, output_path) = test_batch
    image1 = image1.astype(np.float32) / 255
    image2 = image2.astype(np.float32) / 255

    flow = inference_fn(
        image1, image2, input_height=height, input_width=width)
    flow = flow.numpy()

    # Sintel expects horizontal and then vertical flow
    flow = flow[Ellipsis, ::-1]
    conversion_utils.write_flow(output_path, flow)

    if plot_dir and plot_count < num_plots:
      plot_count += 1
      num_rows = 2
      num_columns = 2

      # pylint:disable=cell-var-from-loop
      def subplot_at(column, row):
        plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

      def post_imshow(label):
        plt.xlabel(label)
        plt.xticks([])
        plt.yticks([])

      plt.figure('eval', [10, 10])
      plt.clf()

      subplot_at(0, 0)
      plt.imshow(image1)
      post_imshow(label='Image1')

      subplot_at(1, 0)
      plt.imshow(image2)
      post_imshow(label='Image2')

      subplot_at(0, 1)
      plt.imshow(smurf_plotting.flow_to_rgb(flow))
      post_imshow(label='Prediction')

      plt.subplots_adjust(
          left=0.02,
          bottom=0.02,
          right=1 - 0.02,
          top=1,
          wspace=0.01,
          hspace=0.01)

      filename = 'benchmark_{}.png'.format(plot_count)
      smurf_plotting.save_and_close(os.path.join(plot_dir, filename))
