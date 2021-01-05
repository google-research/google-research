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

"""Utilities to load and process a sign language detection dataset."""
import functools
import os
from typing import Any
from typing import Dict
from typing import Tuple

from pose_format.pose import Pose
from pose_format.pose_header import PoseHeader
from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.tensorflow.pose_body import TF_POSE_RECORD_DESCRIPTION
from pose_format.utils.reader import BufferReader
import tensorflow as tf

from sign_language_detection.args import FLAGS


@functools.lru_cache(maxsize=1)
def get_openpose_header():
  """Get pose header with OpenPose components description."""
  dir_path = os.path.dirname(os.path.realpath(__file__))
  header_path = os.path.join(dir_path, "assets/openpose.poseheader")
  f = open(header_path, "rb")
  reader = BufferReader(f.read())
  header = PoseHeader.read(reader)
  return header


def differentiate_frames(src):
  """Subtract every two consecutive frames."""
  # Shift data to pre/post frames
  pre_src = src[:-1]
  post_src = src[1:]

  # Differentiate src points
  src = pre_src - post_src

  return src


def distance(src):
  """Calculate the Euclidean distance from x:y coordinates."""
  square = src.square()
  sum_squares = square.sum(dim=-1).fix_nan()
  sqrt = sum_squares.sqrt().zero_filled()
  return sqrt


def optical_flow(src, fps):
  """Calculate the optical flow norm between frames, normalized by fps."""

  # Remove "people" dimension
  src = src.squeeze(1)

  # Differentiate Frames
  src = differentiate_frames(src)

  # Calculate distance
  src = distance(src)

  # Normalize distance by fps
  src = src * fps

  return src


minimum_fps = tf.constant(1, dtype=tf.float32)


def load_datum(tfrecord_dict):
  """Convert tfrecord dictionary to tensors."""
  pose_body = TensorflowPoseBody.from_tfrecord(tfrecord_dict)
  pose = Pose(header=get_openpose_header(), body=pose_body)
  tgt = tf.io.decode_raw(tfrecord_dict["is_signing"], out_type=tf.int8)

  fps = pose.body.fps
  frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

  # Get only relevant input components
  pose = pose.get_components(FLAGS.input_components)

  return {
      "fps": pose.body.fps,
      "frames": frames,
      "tgt": tgt,
      "pose_data_tensor": pose.body.data.tensor,
      "pose_data_mask": pose.body.data.mask,
      "pose_confidence": pose.body.confidence,
  }


def process_datum(datum,
                  augment=False):
  """Prepare every datum to be an input-output pair for training / eval.

  Supports data augmentation only including frames dropout.
  Frame dropout affects the FPS, which does change the optical flow.

  Args:
      datum (Dict[str, tf.Tensor]): a dictionary of tensors loaded from the
        tfrecord.
      augment (bool): should apply data augmentation on the datum?

  Returns:
     dict(Dict[str, tf.Tensor]): dictionary including "src" and "tgt" tensors
  """
  masked_tensor = MaskedTensor(
      tensor=datum["pose_data_tensor"], mask=datum["pose_data_mask"])
  pose_body = TensorflowPoseBody(
      fps=datum["fps"], data=masked_tensor, confidence=datum["pose_confidence"])
  pose = Pose(header=get_openpose_header(), body=pose_body)
  tgt = datum["tgt"]

  fps = pose.body.fps
  frames = datum["frames"]

  if augment:
    pose, selected_indexes = pose.frame_dropout(FLAGS.frame_dropout_std)
    tgt = tf.gather(tgt, selected_indexes)

    new_frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

    fps = tf.math.maximum(minimum_fps, (new_frames / frames) * fps)
    frames = new_frames

  flow = optical_flow(pose.body.data, fps)
  tgt = tgt[1:]  # First frame tag is not used

  return {"src": flow, "tgt": tgt}


def prepare_io(datum):
  """Convert dictionary into input-output tuple for Keras."""
  src = datum["src"]
  tgt = datum["tgt"]

  return src, tgt


def batch_dataset(dataset, batch_size):
  """Batch and pad a dataset."""
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={
          "src": [None, None],
          "tgt": [None]
      })

  return dataset.map(prepare_io)


def train_pipeline(dataset):
  """Prepare the training dataset."""
  dataset = dataset.map(load_datum).cache()
  dataset = dataset.repeat()
  dataset = dataset.map(lambda d: process_datum(d, True))
  dataset = dataset.shuffle(FLAGS.batch_size)
  dataset = batch_dataset(dataset, FLAGS.batch_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def test_pipeline(dataset):
  """Prepare the test dataset."""
  dataset = dataset.map(load_datum)
  dataset = dataset.map(process_datum)
  dataset = batch_dataset(dataset, FLAGS.test_batch_size)
  return dataset.cache()


def split_dataset(
    dataset
):
  """Split dataset to train, dev, and test."""

  def is_dev(x, _):
    # Every 3rd item
    return (x + 2) % 4 == 0

  def is_test(x, _):
    # Every 4th item
    return (x + 3) % 4 == 0

  def is_train(x, y):
    return not is_test(x, y) and not is_dev(x, y)

  def recover(_, y):
    return y

  train = train_pipeline(dataset.enumerate().filter(is_train).map(recover))
  dev = test_pipeline(dataset.enumerate().filter(is_dev).map(recover))
  test = test_pipeline(dataset.enumerate().filter(is_test).map(recover))

  return train, dev, test


def get_datasets():
  """Get train, dev, and test datasets."""
  # Set features
  features = {"is_signing": tf.io.FixedLenFeature([], tf.string)}
  features.update(TF_POSE_RECORD_DESCRIPTION)

  # Dataset iterator
  dataset = tf.data.TFRecordDataset(filenames=[FLAGS.dataset_path])
  dataset = dataset.map(
      lambda serialized: tf.io.parse_single_example(serialized, features))

  return split_dataset(dataset)
