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

"""Utils for dataset preparation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import math
import os

from absl import logging
import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow.compat.v2 as tf

import cv2

gfile = tf.io.gfile

feature = tf.train.Feature
bytes_feature = lambda v: feature(bytes_list=tf.train.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
float_feature = lambda v: feature(float_list=tf.train.FloatList(value=v))


def get_example(name, seq, seq_label=None, label_string=None,
                frame_labels_string=None):
  """Returns a single SequenceExample for provided frames and labels in a video.

  There is some replication of information in storing frame_labels_string but
  it is useful to have strings as metadata with each sequence example.

  Also assuming right now index of frame_labels_string and label_string
  refer to classes being listed in frame_labels and label.

  TODO (debidatta): Convert list of label strings to dict.

  Args:
    name: string, name of the sequence.
    seq: dict, dict of list of frames and optionally per-frame labels in video.
    seq_label: int, label of video as an integer.
    label_string: string, label of video as a string.
    frame_labels_string: list, frame level labels as string.
  """
  # Add sequential or frame-level features.
  seq_feats = {}

  if 'video' in seq:
    frames_bytes = [image_to_bytes(frame) for frame in seq['video']]
    seq_feats['video'] = tf.train.FeatureList(feature=frames_bytes)

  # Add per-frame labels.
  if 'labels' in seq:
    label_bytes = [int64_feature([label]) for label in seq['labels']]
    seq_feats['frame_labels'] = tf.train.FeatureList(feature=label_bytes)

  # Create FeatureLists.
  feature_lists = tf.train.FeatureLists(feature_list=seq_feats)

  # Add context or video-level features.
  seq_len = len(seq['video'])
  context_features_dict = {'name': bytes_feature([name.encode()]),
                           'len': int64_feature([seq_len])}

  if seq_label is not None:
    logging.info('Label for %s: %s', name, str(seq_label))
    context_features_dict['label'] = int64_feature([seq_label])

  if label_string:
    context_features_dict['label_string'] = bytes_feature([label_string])

  if frame_labels_string:
    # Store as a single string as all context features should be Features or
    # FeatureLists. Cannot combine types for now.
    labels_string = ','.join(frame_labels_string)
    context_features_dict['framelabels_string'] = bytes_feature([labels_string])
  context_features = tf.train.Features(feature=context_features_dict)

  # Create SequenceExample.
  ex = tf.train.SequenceExample(context=context_features,
                                feature_lists=feature_lists)

  return ex


def write_seqs_to_tfrecords(record_name, name_to_seqs, label,
                            frame_labels_string):
  """Write frames to a TFRecord file."""
  writer = tf.io.TFRecordWriter(record_name)
  for name in name_to_seqs:
    ex = get_example(name, name_to_seqs[name],
                     seq_label=label,
                     frame_labels_string=frame_labels_string)
    writer.write(ex.SerializeToString())
  writer.close()


def image_to_jpegstring(image, jpeg_quality=95):
  """Convert image to a JPEG string."""
  if not isinstance(image, Image.Image):
    raise TypeError('Provided image is not a PIL Image object')
  # This fix to PIL makes sure that we don't get an error when saving large
  # jpeg files. This is a workaround for a bug in PIL. The value should be
  # substantially larger than the size of the image being saved.
  ImageFile.MAXBLOCK = 640 * 512 * 64

  output_jpeg = io.BytesIO()
  image.save(output_jpeg, 'jpeg', quality=jpeg_quality, optimize=True)
  return output_jpeg.getvalue()


def image_to_bytes(image_array):
  """Get bytes formatted image arrays."""
  image = Image.fromarray(image_array)
  im_string = bytes_feature([image_to_jpegstring(image)])
  return im_string


def video_to_frames(video_filename, rotate, fps=0, resize=False,
                    width=224, height=224):
  """Returns all frames from a video.

  Args:
    video_filename: string, filename of video.
    rotate: Boolean: if True, rotates video by 90 degrees.
    fps: Integer, frames per second of video. If 0, it will be inferred from
      metadata of video.
    resize: Boolean, if True resizes images to given size.
    width: Integer, Width of image.
    height: Integer, Height of image.

  Raises:
    ValueError: if fps is greater than the rate of video.
  """
  logging.info('Loading %s', video_filename)
  cap = cv2.VideoCapture(video_filename)

  if fps == 0:
    fps = cap.get(cv2.CAP_PROP_FPS)
    keep_frequency = 1
  else:
    if fps > cap.get(cv2.CAP_PROP_FPS):
      raise ValueError('Cannot sample at a frequency higher than FPS of video')
    keep_frequency = int(float(cap.get(cv2.CAP_PROP_FPS)) / fps)

  frames = []
  timestamps = []
  counter = 0
  if cap.isOpened():
    while True:
      success, frame_bgr = cap.read()
      if not success:
        break
      if counter % keep_frequency == 0:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize:
          frame_rgb = cv2.resize(frame_rgb, (width, height))
        if rotate:
          frame_rgb = cv2.transpose(frame_rgb)
          frame_rgb = cv2.flip(frame_rgb, 1)
        frames.append(frame_rgb)
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
      counter += 1
  return frames, timestamps, fps


def merge_annotations(label, expected_n):
  """Merge annotations from label based on voting."""
  annotations = {}

  for k in range(expected_n):
    segments = np.vstack([label[person_id][str(k)] for person_id in label])
    annotations[k] = np.mean(segments, axis=0)
    # Convert from microseconds to seconds.
    annotations[k] /= 1e6

  sorted_keys = sorted(annotations.keys())
  start_sorted_idxes = np.argsort([annotations[k][0] for k in sorted_keys])
  start_sorted_keys = [sorted_keys[idx] for idx in start_sorted_idxes]

  # Add gaps.
  for i in range(1, expected_n):
    avg_time = 0.5 * (annotations[start_sorted_keys[i-1]][1] +
                      annotations[start_sorted_keys[i]][0])
    annotations[start_sorted_keys[i-1]][1] = avg_time
    annotations[start_sorted_keys[i]][0] = avg_time

  return annotations


def label_timestamps(timestamps, annotations):
  """Each timestamp gets assigned a label according to annotation."""
  labels = []
  sorted_keys = sorted(annotations.keys())
  first_segment = sorted_keys[np.argmin([annotations[k][0]
                                         for k in sorted_keys])]
  last_segment = sorted_keys[np.argmax([annotations[k][1]
                                        for k in sorted_keys])]

  for ts in timestamps:
    assigned = 0
    for k in sorted_keys:
      min_t, max_t = annotations[k]
      # If within the bounds provide the label in annotation.
      if min_t <= ts < max_t:
        labels.append(k)
        assigned = 1
        break
      # If timestamp is higher than last recorded label's timestamp then just
      # copy over the last label
      elif ts >= annotations[last_segment][1]:
        labels.append(last_segment)
        assigned = 1
        break
      # If timestamp is lower than last recorded label's timestamp then just
      # copy over the first label
      elif ts < annotations[first_segment][0]:
        labels.append(first_segment)
        assigned = 1
        break
    # If timestamp was not assigned log a warning.
    if assigned == 0:
      logging.warning('Not able to insert: %s', ts)

  return labels


def create_tfrecords(name, output_dir, input_dir, label_file, input_pattern,
                     files_per_shard, action_label, frame_labels,
                     expected_segments, orig_fps, rotate, resize, width,
                     height):
  """Create TFRecords from videos in a given path.

  Args:
    name: string, name of the dataset being created.
    output_dir: string, path to output directory.
    input_dir: string, path to input videos directory.
    label_file: string, JSON file that contains annotations.
    input_pattern: string, regex pattern to look up videos in directory.
    files_per_shard: int, number of files to keep in each shard.
    action_label: int, Label of actions in video.
    frame_labels: list, list of string describing each class. Class label is
      the index in list.
    expected_segments: int, expected number of segments.
    orig_fps: int, frame rate at which tfrecord will be created.
    rotate: boolean, if True rotate videos by 90 degrees.
    resize: boolean, if True resize to given height and width.
    width: int, Width of frames.
    height: int, Height of frames.
  Raises:
    ValueError: If invalid args are passed.
  """
  if not gfile.exists(output_dir):
    logging.info('Creating output directory: %s', output_dir)
    gfile.makedirs(output_dir)

  if label_file is not None:
    with open(os.path.join(label_file)) as labels_file:
      data = json.load(labels_file)

  if not isinstance(input_pattern, list):
    file_pattern = os.path.join(input_dir, input_pattern)
    filenames = [os.path.basename(x) for x in gfile.glob(file_pattern)]
  else:
    filenames = []
    for file_pattern in input_pattern:
      file_pattern = os.path.join(input_dir, file_pattern)
      filenames += [os.path.basename(x) for x in gfile.glob(file_pattern)]
  filenames = sorted(filenames)
  logging.info('Found %s files', len(filenames))

  names_to_seqs = {}
  num_shards = int(math.ceil(len(filenames)/files_per_shard))
  len_num_shards = len(str(num_shards))
  shard_id = 0
  for i, filename in enumerate(filenames):
    seqs = {}

    frames, video_timestamps, _ = video_to_frames(
        os.path.join(input_dir, filename),
        rotate,
        orig_fps,
        resize=resize,
        width=width,
        height=height)
    seqs['video'] = frames

    if label_file is not None:
      video_id = filename[:-4]
      if video_id in data:
        video_labels = data[video_id]
      else:
        raise ValueError('Video id %s not found in labels file.' % video_id)
      merged_annotations = merge_annotations(video_labels,
                                             expected_segments)

      seqs['labels'] = label_timestamps(video_timestamps, merged_annotations)

    names_to_seqs[os.path.splitext(filename)[0]] = seqs

    if (i + 1) % files_per_shard == 0 or i == len(filenames) - 1:
      output_filename = os.path.join(
          output_dir,
          '%s-%s-of-%s.tfrecord' % (name,
                                    str(shard_id).zfill(len_num_shards),
                                    str(num_shards).zfill(len_num_shards)))
      write_seqs_to_tfrecords(output_filename, names_to_seqs,
                              action_label, frame_labels)
      shard_id += 1
      names_to_seqs = {}
