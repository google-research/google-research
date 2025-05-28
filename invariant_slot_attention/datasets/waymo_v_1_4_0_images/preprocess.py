# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Preprocessing Waymo Open v1.4 for single-frame training."""
import functools
import multiprocessing
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import camera_segmentation_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "read_path", "individual_files/", "Waymo Open v1.4 root path.")
flags.DEFINE_string("write_path", "tfrecords/", "Root write path.")
flags.DEFINE_string("split_name", "training", "Name of dataset split.")
flags.DEFINE_integer(
    "camera_name", open_dataset.CameraName.FRONT,
    ("Camera name, integer between 1 and 5. 1: front, 2: front-left, "
     "3: front-right, 4: side-left, 5: side-right.")
)
flags.DEFINE_bool("require_masks", True, "Require instance segmentation masks.")
flags.DEFINE_integer("n_shards", 16, "Number of write shards.")
flags.DEFINE_string("compression", "zlib", "Compression type.")
flags.DEFINE_integer("image_height", 128, "Image height.")
flags.DEFINE_integer("image_width", 192, "Image width.")


def pad_to_common_shape(label, width = 1280):
  """Add a black bar on top so that the picture height is 1280 pixels."""
  return np.pad(label, [[width - label.shape[0], 0], [0, 0], [0, 0]])


def get_images_and_instance_segmentations_from_frame(frame, camera_name):
  """Preprocess a single camera frame."""
  image = None
  for tmp_image in frame.images:
    if tmp_image.name == camera_name:
      image = tmp_image
  assert image is not None

  image_rgb = tf.image.decode_jpeg(image.image)

  if not image.camera_segmentation_label.panoptic_label:
    # Unlabeled example.
    if FLAGS.require_masks:
      return None
    else:
      # Create a placeholder zero mask.
      shape = image_rgb.shape
      instance_label = np.zeros((shape[0], shape[1], 1), dtype=np.uint8)
      has_mask = False
  else:
    # Read masks, discard semantic segmentation and save instance segmentation.
    panoptic_label = (
        camera_segmentation_utils.decode_single_panoptic_label_from_proto(
            image.camera_segmentation_label))
    _, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(  # pylint: disable=line-too-long
        panoptic_label, image.camera_segmentation_label.panoptic_label_divisor)
    has_mask = True

  # Pad side-left and side-right images.
  if FLAGS.camera_name in [
      open_dataset.CameraName.SIDE_LEFT, open_dataset.CameraName.SIDE_RIGHT
  ]:
    image_rgb = pad_to_common_shape(image_rgb)
    instance_label = pad_to_common_shape(instance_label)

  image_rgb = tf.cast(
      tf.image.resize(
          tf.cast(image_rgb, tf.float32) / 255.,
          [FLAGS.image_height, FLAGS.image_width],
          method="bilinear") * 255., tf.uint8)

  instance_label = tf.cast(
      tf.image.resize(
          instance_label.astype(np.uint8),
          [FLAGS.image_height, FLAGS.image_width],
          method="nearest"), tf.uint8)

  return image_rgb, instance_label, has_mask


def parse_waymo_example(serialized_example):
  """Unserializes a serialized Waymo Open example."""
  frame = open_dataset.Frame()
  frame.ParseFromString(bytearray(serialized_example))

  out = get_images_and_instance_segmentations_from_frame(
      frame, FLAGS.camera_name)

  if out is None:
    return None

  image, mask, has_mask = out

  return {
      "image": image,
      "segmentations": mask,
      "has_mask": has_mask,
      "camera": FLAGS.camera_name
  }


def _to_list(x):
  return np.array(x).flatten().tolist()


def _to_bytes(value, dtype=np.uint8):
  return np.array(value, dtype=dtype).tobytes()


def _bytes_feature_from_numpy(x, pool):
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=pool.map(_to_bytes, _to_list(x))))


def _int64_feature_from_numpy(x):
  return tf.train.Feature(int64_list=tf.train.Int64List(
      value=_to_list(x)))


def _float_feature_from_tensor(tensor):
  return tf.train.Feature(float_list=tf.train.FloatList(value=_to_list(tensor)))


def serialize_example(example, pool):
  """Serializes a Waymo Open example."""
  num_objects = len(np.unique(example["segmentations"]))
  feature = {
      "image":
          _bytes_feature_from_numpy(example["image"], pool),
      "segmentations":
          _bytes_feature_from_numpy(example["segmentations"], pool),
      "num_objects":
          _int64_feature_from_numpy([num_objects]),
      "has_mask":
          _int64_feature_from_numpy([int(example["has_mask"])]),
      "camera":
          _int64_feature_from_numpy(example["camera"])
  }
  return tf.train.Example(features=tf.train.Features(
      feature=feature)).SerializeToString()


def main(_):
  # Used to serialize images faster.
  pool = multiprocessing.Pool(8)

  data_path = os.path.join(FLAGS.read_path, FLAGS.split_name,
                           "segment-*_with_camera_labels.tfrecord")
  files = tf.data.Dataset.list_files(data_path)
  data_reader = functools.partial(
      tf.data.TFRecordDataset, buffer_size=10 * (2**20))  # 10 MB
  dataset = files.interleave(
      data_reader, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  iterator = dataset.as_numpy_iterator()

  base_write_path = os.path.join(FLAGS.write_path, FLAGS.split_name,
                                 "camera_{:d}".format(FLAGS.camera_name))

  if not os.path.isdir(base_write_path):
    os.makedirs(base_write_path)

  out_file = "waymo_open_%sx%s.tfrecords-%s" % (
      FLAGS.image_height, FLAGS.image_width, FLAGS.compression,
  )
  out_path = os.path.join(base_write_path, out_file)

  out_file_pattern = out_path + "-%05d-of-%05d"
  out_filenames = [
      out_file_pattern % (i, FLAGS.n_shards) for i in range(FLAGS.n_shards)
  ]

  options = tf.io.TFRecordOptions(compression_type=FLAGS.compression.upper())
  writers = [tf.io.TFRecordWriter(f, options=options) for f in out_filenames]

  saved = 0
  without_labels = 0
  skipped = 0

  for s_example in iterator:
    example = parse_waymo_example(s_example)

    if example is None:
      # No segmentation mask in example, but we require it.
      without_labels += 1
      skipped += 1
      continue

    if not example["has_mask"]:
      # No segmentation mask in example and we do not require it.
      without_labels += 1

    shard_idx = saved % FLAGS.n_shards
    writer = writers[shard_idx]

    features = serialize_example(example, pool)
    writer.write(features)

    saved += 1

    if saved % 1000 == 0:
      logging.info(
          "%d saved, %d without labels, %d skipped",
          saved, without_labels, skipped)

  pool.close()
  pool.join()


if __name__ == "__main__":
  app.run(main)
