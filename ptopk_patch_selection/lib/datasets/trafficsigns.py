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

# Lint as: python3
"""Swedish Trafic Signs dataset.

This is a subset of the data from the Swedish Traffic Sign dataset at
http://www.isy.liu.se/cvl/research/trafficSigns/


It follows the filtering and preprocessing from the paper
  Katharopoulos & Fleuret,
  "Processing Megapixel Images with Deep Attention-Sampling Models"
  ICML 2019, https://arxiv.org/abs/1905.03711

And contains 747 training and 684 test images.
"""
import collections
import os
import pathlib

import tensorflow as tf


def load(split: str) -> tf.data.Dataset:
  """Loads the traffic-sign dataset.

  Arguments:
    split: one of "train" or "test"

  Returns:
    tf.data.Dataset
  """

  def _parse_single_example(example):
    feature_description = {
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)
    img = tf.io.decode_jpeg(features["image"], channels=3)
    img = tf.reshape(img, [960, 1280, 3])  # For shape inference.
    features["image"] = img
    return features

  split = split.lower()
  assert split in ("train", "test"), f"Invalid split: {split}"

  base_dir = pathlib.Path(__file__).parent / "trafficsigns"
  filename = "set1.tfrecords" if split == "train" else "set2.tfrecords"
  path = os.path.join(base_dir, filename)

  data = tf.data.TFRecordDataset(path)
  return data.map(_parse_single_example,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()


##############################################################################
# The following functions were used to convert the original files into
# TFRecord files. They are for reproducibility only.
##############################################################################


def _load_original_dataset(data_directory, setname):
  """Loads a part of the trafficSigns dataset.

  The following code assumes that these files have been downloaded and extracted
  into data_directory.

  http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/Set1Part0.zip
  http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/annotations.txt
  http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/Set2Part0.zip
  http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/annotations.txt

  Args:
    data_directory: Directory containing the downloaded files.
    setname: The dataset to work on, Either "set1" or "set2".

  Returns:
    A pair of lists consisting of filepaths and the found signs.
  """

  Sign = collections.namedtuple("Sign", ["visibility", "type", "name"])
  data_directory = pathlib.Path(data_directory)
  filename = data_directory / setname / "annotations.txt"
  with tf.io.gfile.GFile(filename) as f:
    files, annotations = zip(*(l.strip().split(":", 1) for l in f))

  all_signs = []
  for annotation in annotations:
    signs = []
    for sign in annotation.split(";"):
      if sign == [""] or not sign: continue
      parts = [s.strip() for s in sign.split(",")]
      if parts[0] == "MISC_SIGNS": continue
      signs.append(Sign(parts[0], parts[5], parts[6]))
    all_signs.append(signs)

  filepaths = (data_directory / setname / f for f in files)
  return zip(filepaths, all_signs)


def _preprocess_and_filter_original_dataset(data):
  """Filters out badly visible or uninteresting signs."""

  label_order = ("EMPTY", "50_SIGN", "70_SIGN", "80_SIGN")

  filtered_data = []
  for image, signs in data:
    if not signs:
      filtered_data.append((image, label_order.index("EMPTY")))
    else:
      # take the most visible of the interesting signs
      signs = [s for s in signs
               if s.name in label_order and s.visibility == "VISIBLE"]
      if signs:
        filtered_data.append((image, label_order.index(signs[0].name)))
  return filtered_data


def _create_tfrecords_file(data, output_file):
  """Writes File to TFRecords file."""
  with tf.io.TFRecordWriter(str(output_file)) as writer:
    for image, sign in data:
      with tf.io.gfile.GFile(image, "rb") as f:
        image_string = f.read()
      feature = {
          "image": tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[image_string])),
          "label": tf.train.Feature(
              int64_list=tf.train.Int64List(value=[sign]))
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())


def _create_tfrecords_dataset(data_directory, output_directory):
  """Creates TFRecord files."""

  data_directory = pathlib.Path(data_directory)
  output_directory = pathlib.Path(output_directory)

  for setname in ["set1", "set2"]:
    data = _load_original_dataset(data_directory, setname)
    filtered_data = _preprocess_and_filter_original_dataset(data)
    output_file = output_directory / (setname.lower() + ".tfrecords")
    _create_tfrecords_file(filtered_data, output_file)


if __name__ == "__main__":
  import urllib.request  # pylint: disable=g-import-not-at-top
  import zipfile  # pylint: disable=g-import-not-at-top
  import ssl  # pylint: disable=g-import-not-at-top

  # A quick hack to get the urllib.request to ignore invalid SSL certs
  ssl._create_default_https_context = ssl._create_unverified_context  # pylint: disable=protected-access

  target_dir = directory = pathlib.Path(__file__).parent / "trafficsigns"
  if not target_dir.exists():
    target_dir.mkdir()
  print("starting downloads", flush=True)
  if not (target_dir / "set1.zip").exists():
    urllib.request.urlretrieve(
        "https://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/Set1Part0.zip",
        target_dir / "set1.zip")

    urllib.request.urlretrieve(
        "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/Set2Part0.zip",
        target_dir / "set2.zip")

    print("files downloaded, starting extraction", flush=True)

    zipfile.ZipFile(target_dir / "set1.zip").extractall(target_dir / "set1")
    zipfile.ZipFile(target_dir / "set2.zip").extractall(target_dir / "set2")

    urllib.request.urlretrieve(
        "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/annotations.txt",
        target_dir / "set1" / "annotations.txt")
    urllib.request.urlretrieve(
        "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/annotations.txt",
        target_dir / "set2" / "annotations.txt")
  print("files extracted, creating TFRecords.", flush=True)
  _create_tfrecords_dataset(target_dir, target_dir)
  print("all done")
