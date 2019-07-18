# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Convert list of videos to tfrecords based on SequenceExample."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import math
import os
import random

from absl import app
from absl import flags
from absl import logging

from dataset_utils import label_timestamps
from dataset_utils import merge_annotations
from dataset_utils import write_seqs_to_tfrecords
import scipy.io as sio

import cv2

flags.DEFINE_string('dir', None, 'Path to videos.')
flags.DEFINE_string('name', None, 'Name of the dataset being created. This will'
                    'be used as a prefix.')
flags.DEFINE_string('vid_list', None, 'Path to list of folders with frames of '
                    'videos.')
flags.DEFINE_string('extension', 'jpg', 'Extension of images.')
flags.DEFINE_string(
    'label_file', None, 'Provide a corresponding labels file'
    'that stores per-frame or per-sequence labels.')
flags.DEFINE_string('output_dir', '/tmp/tfrecords/', 'Output directory where'
                    'tfrecords will be stored.')
flags.DEFINE_integer('vids_per_shard', 1, 'Number of videos to store in a'
                     'shard.')
flags.DEFINE_list(
    'frame_labels', '', 'Comma separated list of descriptions '
    'for labels given on a per frame basis. For example: '
    'winding_up,early_cocking,acclerating,follow_through')
flags.DEFINE_integer('action_label', -1, 'Action label of all videos.')
flags.DEFINE_integer('expected_segments', -1, 'Expected number of segments.')
flags.DEFINE_boolean('rotate', False, 'Rotate videos by 90 degrees before'
                     'creating tfrecords')
flags.DEFINE_boolean('resize', True, 'Resize videos to a given size.')
flags.DEFINE_integer('width', 224, 'Width of frames in the TFRecord.')
flags.DEFINE_integer('height', 224, 'Height of frames in the TFRecord.')
flags.DEFINE_integer('fps', 30, 'Frames per second in video.')

flags.mark_flag_as_required('name')
flags.mark_flag_as_required('dir')
flags.mark_flag_as_required('vid_list')
FLAGS = flags.FLAGS


def preprocess(im, rotate, resize, width, height):
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  if resize:
    im = cv2.resize(im, (width, height))
  if rotate:
    im = cv2.transpose(im)
    im = cv2.flip(im, 1)
  return im


def get_frames_in_folder(path, rotate, resize, width, height):
  """Returns all frames from a video in a given folder.

  Args:
    path: string, directory containing frames of a video.
    rotate: Boolean, if True rotates an image by 90 degrees.
    resize: Boolean, if True resizes images to given size.
    width: Integer, Width of image.
    height: Integer, Height of image.
  Returns:
    frames: list, list of frames in a  video.
  Raises:
    ValueError: When provided directory doesn't exist.
  """
  if not os.path.isdir(path):
    raise ValueError('Provided path %s is not a directory' % path)
  else:
    im_list = sorted(glob.glob(os.path.join(path, '*.%s' % FLAGS.extension)))

  frames = [preprocess(cv2.imread(im), rotate, resize, width, height)
            for im in im_list]
  return frames


def get_name(filename, videos_dir, penn_action=False):
  """Add label to name for Penn Action dataset."""
  if penn_action:
    labels_path = os.path.join(videos_dir, 'labels', '%s.mat' % filename)
    annotation = sio.loadmat(labels_path)
    label = annotation['action'][0]
    return '{}_{}'.format(filename, label)
  else:
    return filename


def get_timestamps(frames, fps, offset=0.0):
  """Returns timestamps for frames in a video."""
  return [offset + x/float(fps) for x in xrange(len(frames))]


def create_tfrecords(name, output_dir, videos_dir, vid_list, label_file,
                     frame_labels, fps, expected_segments):
  """Create TFRecords from videos in a given path.

  Args:
    name: string, name of the dataset being created.
    output_dir: string, path to output directory.
    videos_dir: string, path to input videos directory.
    vid_list: string, path to file containing list of folders where frames
      are stored.
    label_file: string, JSON file that contains annotations.
    frame_labels: list, list of string describing each class. Class label is
      the index in list.
    fps: integer, frames per second with which the images were extracted.
    expected_segments: int, expected number of segments.
  Raises:
    ValueError: If invalid args are passed.
  """
  if not os.path.exists(output_dir):
    logging.info('Creating output directory: %s', output_dir)
    os.makedirs(output_dir)

  with open(vid_list, 'r') as f:
    paths = [os.path.join(videos_dir, x.strip()) for x in f.readlines()]

  random.shuffle(paths)

  if label_file is not None:
    with open(os.path.join(label_file)) as labels_file:
      data = json.load(labels_file)

  names_to_seqs = {}
  num_shards = int(math.ceil(len(paths)/FLAGS.vids_per_shard))
  len_num_shards = len(str(num_shards))
  shard_id = 0
  for i, path in enumerate(paths):
    seq = {}

    vid_name = get_name(os.path.basename(path), videos_dir)
    frames = get_frames_in_folder(path, FLAGS.rotate, FLAGS.resize,
                                  FLAGS.width, FLAGS.height)
    seq['video'] = frames

    if label_file is not None:
      video_id = os.path.basename(path)
      if video_id in data:
        video_labels = data[video_id]
      else:
        raise ValueError('Video id %s not found in labels file.' % video_id)

      merged_annotations = merge_annotations(video_labels,
                                             expected_segments)
      video_timestamps = get_timestamps(frames, fps)
      seq['labels'] = label_timestamps(video_timestamps, merged_annotations)

    names_to_seqs[vid_name] = seq
    if (i + 1) % FLAGS.vids_per_shard == 0 or i == len(paths)-1:
      output_filename = os.path.join(
          output_dir,
          '%s-%s-of-%s.tfrecord' % (name,
                                    str(shard_id).zfill(len_num_shards),
                                    str(num_shards).zfill(len_num_shards)))
      write_seqs_to_tfrecords(output_filename, names_to_seqs,
                              FLAGS.action_label, frame_labels)

      shard_id += 1
      names_to_seqs = {}


def main(_):
  create_tfrecords(FLAGS.name, FLAGS.output_dir, FLAGS.dir, FLAGS.vid_list,
                   FLAGS.label_file, FLAGS.frame_labels, FLAGS.fps,
                   FLAGS.expected_segments)


if __name__ == '__main__':
  app.run(main)
