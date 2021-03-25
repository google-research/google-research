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

"""Synthetic dataset of billiard images with on-the-fly label generation."""

import json
import pathlib
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image
import tensorflow as tf


_SPLITS = ["train", "valid", "test"]


def load_billiard(split, label_fn):
  """Creates a dataset with billiard images and computed labels.

  Args:
    split: One of "train", "valid" or "test".
    label_fn: String defining the label function of the image.

  Returns:
    A dataset and the number of classes for the task.

  Raises:
    RuntimeError: on IO errors.
  """
  if split not in _SPLITS:
    raise ValueError(f"Unknown split '{split}', should be one of {_SPLITS}.")

  directory = pathlib.Path(__file__).parent / "billiard" / split
  if not directory.exists():
    raise RuntimeError("Billiard dataset not found. See README for details.")

  label_fn, num_classes = {
      "rightmost-number": (rightmost_number, 9),
      "max-left-right": (max_left_right, 9),
      "sum-all-modulo10": (sum_all_modulo10, 10),
      "left-color-min-max": (left_color_min_max, 9),
  }[label_fn]

  dataset = _generator_billiard_dataset(directory,
                                        label_fn,
                                        cache=True)

  return dataset, num_classes


def _load_png_as_numpy(path):
  with tf.io.gfile.GFile(path, "rb") as image_file:
    image = Image.open(image_file)
    image = np.array(image, dtype="uint8")[Ellipsis, :3]
    return image


def _generator_billiard_dataset(directory,
                                label_fn,
                                offset = None,
                                length = None,
                                cache = True):
  """Builds a generator yielding samples for `tf.data.Dataset.from_generator`.

  Args:
    directory: Path to the directory containing images in PNG format and
      associated data in JSON.
    label_fn: A function that produce an integer label from the JSON data.
    offset: Start reading (sorted) files at this index. Default will be 0.
    length: Number of files to read (samples). Default will be all.
    cache: Should the dataset be cached in memory.

  Returns:
    The dataset.
  """
  directory = pathlib.Path(directory)
  files = [x.name for x in directory.iterdir()]
  ids = sorted([int(f[:-len(".png")]) for f in files if ".png" in f])
  image_for_shape = _load_png_as_numpy(directory / f"{ids[0]}.png")

  offset = offset or 0
  length = length or (len(ids) - offset)
  assert length > 0 and offset >= 0 and offset + length <= len(ids)
  ids = ids[offset:offset+length]

  types = {"image": tf.uint8, "label": tf.int32}
  shapes = {"image": tf.TensorShape(image_for_shape.shape),
            "label": tf.TensorShape([])}

  def generator():
    for sample_id in ids:
      with tf.io.gfile.GFile(directory / f"{sample_id}.json") as json_file:
        data = json.load(json_file)

      image = _load_png_as_numpy(directory / f"{sample_id}.png")

      label = label_fn(data)
      yield {"image": image, "label": label}

  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  if cache:
    dataset = dataset.cache()

  return dataset


def rightmost_number(objects):
  rightmost_object = max(objects, key=lambda o: o["position"][0])
  return rightmost_object["number"] - 1


def max_left_right(objects):
  leftmost_object = min(objects, key=lambda o: o["position"][0])
  rightmost_object = max(objects, key=lambda o: o["position"][0])
  return max(leftmost_object["number"], rightmost_object["number"]) - 1


def sum_all_modulo10(objects):
  return sum(o["number"] for o in objects) % 10


def left_color_min_max(objects):
  """Returns min or max number on the balls depending on the leftmost color."""
  min_color = ["green", "blue", "purple"]
  max_color = ["yellow", "red", "orange"]
  leftmost_color = min(objects, key=lambda o: o["position"][0])["color"]
  if leftmost_color in min_color:
    min_or_max = min
  elif leftmost_color in max_color:
    min_or_max = max
  else:
    raise ValueError(f"Color '{leftmost_color}' is not supported.")
  return min_or_max(o["number"] for o in objects) - 1


