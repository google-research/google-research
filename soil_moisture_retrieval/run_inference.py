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

"""Script to run inference on soil moisture models released with the paper."""

import itertools
from typing import Any, Dict

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

_EXPORTED_MODEL_DIR_FLAG = flags.DEFINE_string(
    "exported_model_dir", "", "Path to the exported model directory.")
_INPUT_DATA_PATH_FLAG = flags.DEFINE_string(
    "input_data_path", "",
    "Path to input TFRecord file used to fetch samples to run inference on.")
_NUM_INFERENCE_SAMPLES_FLAG = flags.DEFINE_integer(
    "num_inference_samples", 5,
    "The number of samples from input_data_path to run inference on.")
_NUM_SKIP_SAMPLES_FLAG = flags.DEFINE_integer(
    "num_skip_samples", 0,
    "The number of samples to skip from input_data_path after which inference starts."
)

_INPUT_IMAGE_KEY = "ImageTensor:0"
_LOW_RES_INPUT_IMAGE_KEY = "LowResImageTensor:0"
_AUXILIARY_INPUT_KEY = "AuxTensor:0"
# This is the output key that provides us the soil moisture value after it is
# clipped to a [0., 1.] range during inference.
_OUTPUT_KEY = "clip_by_value:0"

_SENSOR_ID = "sensor_id"

# Placeholder value for datapoints where data isn't available.
_NO_DATA_VALUE = 0.
_NO_DATA_VALUE_INT = int(_NO_DATA_VALUE)

_SENTINEL_1_VV = "sentinel_1_VV"
_SENTINEL_1_VH = "sentinel_1_VH"
_SENTINEL_1_ANGLE = "sentinel_1_angle"

_SENTINEL_2_B2 = "sentinel_2_B2"
_SENTINEL_2_B3 = "sentinel_2_B3"
_SENTINEL_2_B4 = "sentinel_2_B4"
_SENTINEL_2_B5 = "sentinel_2_B5"
_SENTINEL_2_B6 = "sentinel_2_B6"
_SENTINEL_2_B7 = "sentinel_2_B7"
_SENTINEL_2_B8 = "sentinel_2_B8"
_SENTINEL_2_B11 = "sentinel_2_B11"
_SENTINEL_2_B12 = "sentinel_2_B12"

_NASA_DEM_ELEVATION = "nasa_dem_elevation"

_SOIL_GRIDS_SAND = "soil_grids_sand_sand_0-5cm_mean"
_SOIL_GRIDS_SILT = "soil_grids_silt_silt_0-5cm_mean"
_SOIL_GRIDS_CLAY = "soil_grids_clay_clay_0-5cm_mean"
_SOIL_GRIDS_BDOD = "soil_grids_bd_bdod_0-5cm_mean"

_GLDAS_SM = "gldas_sm"

_SMAP_SM = "smap_sm"


def parse_example(serialized_example):
  """Parses a serialized TFExample.

  Args:
    serialized_example: Serialized data (tfrecord).

  Returns:
    parsed_example: Parsed datapoint as dict.
  """

  string_features = [_SENSOR_ID]
  float_features = [
      "center_lat", "center_long", "timestamp", "sm_0_5", "sm_5_5"
  ]
  int_features = [_SMAP_SM, _GLDAS_SM]

  # Image features
  sen_1_features = [_SENTINEL_1_VV, _SENTINEL_1_VH, _SENTINEL_1_ANGLE]
  sen_2_features = [
      _SENTINEL_2_B2,
      _SENTINEL_2_B3,
      _SENTINEL_2_B4,
      _SENTINEL_2_B5,
      _SENTINEL_2_B6,
      _SENTINEL_2_B7,
      _SENTINEL_2_B8,
      _SENTINEL_2_B11,
      _SENTINEL_2_B12,
  ]
  nasa_dem_features = [_NASA_DEM_ELEVATION]
  soil_grids_features = [
      _SOIL_GRIDS_SAND, _SOIL_GRIDS_SILT, _SOIL_GRIDS_CLAY, _SOIL_GRIDS_BDOD
  ]
  image_features = (
      sen_1_features + sen_2_features + nasa_dem_features + soil_grids_features)

  keys_to_features = {}
  for feature_list, tf_feature_type in zip(
      [string_features, float_features, int_features, image_features], [
          tf.io.FixedLenFeature([], tf.string),
          tf.io.FixedLenFeature([], tf.float32, default_value=_NO_DATA_VALUE),
          tf.io.FixedLenFeature([], tf.int64, default_value=_NO_DATA_VALUE_INT),
          tf.io.FixedLenFeature((512, 512), tf.int64)
      ]):
    keys_to_features = {
        **keys_to_features,
        **dict(zip(feature_list, itertools.repeat(tf_feature_type)))
    }

  parsed_example = tf.io.parse_single_example(
      serialized=serialized_example, features=keys_to_features)
  return parsed_example


def decode_images_in_example(example):
  """Decodes the images and scalars in the parsed example.

  The decoder decodes the parsed example into a format that the underlying
  soil moisture model understands.

  Args:
    example: Parsed example as dict

  Returns:
    example: Example with parsed image, low_res_image, image_name, timestamp,
      auxiliary_inputs and label tensors.
  """
  decoded_example = {}

  # Normalize the DEM image. Our dataset has unnormalized DEM imagery.
  elevation = tf.cast(example[_NASA_DEM_ELEVATION], dtype=tf.float32)
  min_elevation = 0.0
  max_elevation = 3000.0
  example[_NASA_DEM_ELEVATION] = ((elevation - min_elevation) /
                                  (max_elevation - min_elevation)) * 255.0
  example[_NASA_DEM_ELEVATION] = tf.cast(
      tf.clip_by_value(example[_NASA_DEM_ELEVATION], 0.0, 255.0),
      dtype=tf.int64)

  def _combine_bands_into_image(bands):
    filtered_bands = bands
    if len(filtered_bands) > 1:
      image = tf.concat(
          [tf.expand_dims(example[band], axis=-1) for band in filtered_bands],
          axis=-1)
    else:
      image = tf.expand_dims(example[filtered_bands[0]], axis=-1)

    return image

  image = _combine_bands_into_image(
      (_SENTINEL_1_VV, _SENTINEL_1_VH, _SENTINEL_1_ANGLE, _SENTINEL_2_B4,
       _SENTINEL_2_B3, _SENTINEL_2_B2, _SENTINEL_2_B8, _SENTINEL_2_B11,
       _SENTINEL_2_B12, _NASA_DEM_ELEVATION))
  decoded_example["image"] = tf.cast(image, dtype=tf.float32)

  low_res_image = _combine_bands_into_image(
      (_SOIL_GRIDS_SAND, _SOIL_GRIDS_SILT,
       _SOIL_GRIDS_CLAY, _SOIL_GRIDS_BDOD))
  decoded_example["low_res_image"] = tf.cast(low_res_image, dtype=tf.float32)

  decoded_example["image_name"] = example[_SENSOR_ID]
  decoded_example["timestamp"] = example["timestamp"]
  decoded_example["label"] = tf.cond(
      pred=tf.equal(example["sm_0_5"], _NO_DATA_VALUE),
      true_fn=lambda: example["sm_5_5"],
      false_fn=lambda: example["sm_0_5"])

  decoded_example["auxiliary_inputs"] = tf.stack([
      tf.cast(example[_SMAP_SM], dtype=tf.float32) / 255.0,
      tf.cast(example[_GLDAS_SM], dtype=tf.float32) / 255.0
  ])

  return decoded_example


def main(_):
  tf.reset_default_graph()

  with tf.Session() as sess:
    files = tf.io.gfile.glob(_INPUT_DATA_PATH_FLAG.value)
    tf_files = tf.convert_to_tensor(files)
    tf_record_data = tf.data.TFRecordDataset(tf_files, compression_type="GZIP")

    # Parses and decodes the TFExamples present in the dataset.
    tf_record_data = tf_record_data.map(parse_example, num_parallel_calls=10)
    tf_record_data = tf_record_data.map(
        decode_images_in_example, num_parallel_calls=10)

    data_iter = tf.data.make_initializable_iterator(tf_record_data)
    # Initialize the dataset.
    print("Initializing the dataset.")
    sess.run(data_iter.initializer)
    data_iter = data_iter.get_next()

    # Skip samples if needed.
    for _ in range(_NUM_SKIP_SAMPLES_FLAG.value):
      sample = sess.run(data_iter)

    # Load the saved model.
    print("Loading the saved model.")
    tf.saved_model.load(sess, {"serve"}, _EXPORTED_MODEL_DIR_FLAG.value)

    # Run inference on samples.
    for idx in range(_NUM_INFERENCE_SAMPLES_FLAG.value):
      sample = sess.run(data_iter)
      # Inputs are cropped/padded to a 256x256 size within the model
      # automatically.
      # We batch the inputs since the model expects a batch of size 1 for
      # inference.
      image_batch = np.expand_dims(sample["image"], axis=0)
      low_res_image_batch = np.expand_dims(sample["low_res_image"], axis=0)
      auxiliary_inputs_batch = np.expand_dims(
          sample["auxiliary_inputs"], axis=0)
      # Run the model on the inputs and obtain the soil moisture prediction.
      raw_outputs = sess.run(
          _OUTPUT_KEY,
          feed_dict={
              _INPUT_IMAGE_KEY: image_batch,
              _LOW_RES_INPUT_IMAGE_KEY: low_res_image_batch,
              _AUXILIARY_INPUT_KEY: auxiliary_inputs_batch
          })
      print(f"Prediction {idx}:", raw_outputs.squeeze())


if __name__ == "__main__":
  app.run(main)
