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

"""Preprocess CLEVRTex."""
import json
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_dirs", 50, "Total number of subdirectories.")
flags.DEFINE_integer("n_shards", 64,
                     "Total number of shards. We spawn one worker per shard.")
flags.DEFINE_integer("shard_idx", 0, "Shard index this process will generate.")
flags.DEFINE_string("name", "full", "Name of the CLEVRTex subset.")
flags.DEFINE_string("base_read_path", "download",
                    "Base read path. Should have upacked CLEVRTex folders.")
flags.DEFINE_string("base_write_path", "tfrecords", "Base write path.")
flags.DEFINE_string("compression", "zlib", "Compression type.")


def _to_list(x):
  return np.array(x).flatten().tolist()


def _to_bytes(value, dtype=np.uint8):
  return np.array(value, dtype=dtype).tobytes()


def _bytes_feature_from_numpy(x):
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=map(_to_bytes, _to_list(x))))


def _int64_feature_from_numpy(x):
  return tf.train.Feature(int64_list=tf.train.Int64List(
      value=_to_list(x)))


def _float_feature_from_numpy(x):
  return tf.train.Feature(float_list=tf.train.FloatList(
      value=_to_list(x)))


def load_image(image_path):
  with open(image_path, "rb") as f:
    return np.array(Image.open(f))


def load_json(json_path):
  with open(json_path, "r") as f:
    return json.load(f)


def process_example(image, segmentations, depth, albedo, normal, shadow,
                    metadata, max_n_entities):
  """Preprocess CLEVRTex example."""
  object_3d_coords = np.zeros((max_n_entities, 3), dtype=np.float32)
  object_pixel_coords = np.zeros((max_n_entities, 3), dtype=np.float32)
  object_rotations = np.zeros((max_n_entities,), dtype=np.float32)
  for d in metadata["objects"]:
    index = d["index"] - 1  # CLEVRTex indices start at 1
    object_3d_coords[index] = d["3d_coords"]
    object_pixel_coords[index] = d["pixel_coords"]
    object_rotations[index] = d["rotation"]
  # image, albedo and normal have four channels.
  return (image[Ellipsis, :-1], segmentations, depth, albedo[Ellipsis, :-1],
          normal[Ellipsis, :-1], shadow, metadata["num_objects"],
          metadata["image_index"], object_3d_coords,
          object_pixel_coords, object_rotations)


def serialize_example(image, segmentations, depth, albedo, normal, shadow,
                      num_objects, image_index, object_3d_coords,
                      object_pixel_coords, object_rotations):
  """Serialize CLEVRTex example."""
  feature = {
      "image": _bytes_feature_from_numpy(image),
      "segmentations": _bytes_feature_from_numpy(segmentations),
      "depth": _bytes_feature_from_numpy(depth),
      "albedo": _bytes_feature_from_numpy(albedo),
      "normal": _bytes_feature_from_numpy(normal),
      "shadow": _bytes_feature_from_numpy(shadow),
      "num_objects": _int64_feature_from_numpy([num_objects]),
      "image_index": _int64_feature_from_numpy([image_index]),
      "object_3d_coords": _float_feature_from_numpy(object_3d_coords),
      "object_pixel_coords": _float_feature_from_numpy(object_pixel_coords),
      "object_rotations": _float_feature_from_numpy(object_rotations)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
  n_dirs = FLAGS.n_dirs
  n_shards = FLAGS.n_shards
  shard_idx = FLAGS.shard_idx
  name = FLAGS.name
  compression = FLAGS.compression
  base_read_path = os.path.join(FLAGS.base_read_path,
                                "clevrtex_{:s}".format(name))
  base_write_path = os.path.join(FLAGS.base_write_path,
                                 "clevrtex_{:s}".format(name))

  n_files_per_dir = 1000
  image_size = [240, 320]
  max_n_entities = 11

  if not os.path.isdir(base_write_path):
    os.makedirs(base_write_path)

  out_file = "clevrtex_%s_%sx%s.tfrecords-%s" % (
      name, image_size[0], image_size[1], compression,
  )
  out_path = os.path.join(base_write_path, out_file)
  logging.info("out_path: {:s}".format(out_path))

  out_file_pattern = out_path + "-%05d-of-%05d"
  out_filename = out_file_pattern % (shard_idx, n_shards)

  options = tf.io.TFRecordOptions(compression_type=compression.upper())
  writer = tf.io.TFRecordWriter(out_filename, options=options)

  try:
    for dir_idx in range(n_dirs):
      logging.info("dir {:d}/{:d}".format(dir_idx, n_dirs))
      for file_idx in range(n_files_per_dir):

        example_idx = dir_idx * 1000 + file_idx
        writer_idx = example_idx % n_shards

        if file_idx % 100 == 0:
          logging.info("file {:d}/{:d}".format(file_idx, n_files_per_dir))

        # We spawn a separate worker for each shard.
        # Otherwise, the preprocessing takes too long.
        if writer_idx != shard_idx:
          continue

        image_path = os.path.join(
            base_read_path, str(dir_idx),
            "CLEVRTEX_{:s}_{:06d}.png".format(name, example_idx))
        segmentations_path = os.path.join(
            base_read_path, str(dir_idx),
            "CLEVRTEX_{:s}_{:06d}_flat.png".format(name, example_idx))
        depth_path = os.path.join(
            base_read_path, str(dir_idx),
            "CLEVRTEX_{:s}_{:06d}_depth_0001.png".format(name, example_idx))
        albedo_path = os.path.join(
            base_read_path, str(dir_idx),
            "CLEVRTEX_{:s}_{:06d}_albedo_0001.png".format(name, example_idx))
        normal_path = os.path.join(
            base_read_path, str(dir_idx),
            "CLEVRTEX_{:s}_{:06d}_normal_0001.png".format(name, example_idx))
        shadow_path = os.path.join(
            base_read_path, str(dir_idx),
            "CLEVRTEX_{:s}_{:06d}_shadow_0001.png".format(name, example_idx))

        image = load_image(image_path)
        segmentations = load_image(segmentations_path)
        depth = load_image(depth_path)
        albedo = load_image(albedo_path)
        normal = load_image(normal_path)
        shadow = load_image(shadow_path)

        json_path = os.path.join(
            base_read_path, str(dir_idx),
            "CLEVRTEX_{:s}_{:06d}.json".format(name, example_idx))
        metadata = load_json(json_path)

        features = serialize_example(*process_example(
            image, segmentations, depth, albedo, normal, shadow, metadata,
            max_n_entities)).SerializeToString()
        writer.write(features)

  finally:
    writer.close()


if __name__ == "__main__":
  app.run(main)
