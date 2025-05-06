# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Utilities to load and save data."""

import functools
import json
import os
import random
from typing import Optional

from layout_gvt_public.datasets import datasets_info
import numpy as np
import tensorflow as tf


# Default discretization values.
DEFAULT_RESOLUTION_WIDTH = 32
DEFAULT_RESOLUTION_HEIGHT = 32


def _clamp(value):
  """Truncates `value` to the [0, 1] range."""
  return np.clip(value, 0.0, 1.0)


def _normalize_entries(documents, shuffle=False):
  """Normalizes the bounding box annotations to the [0, 1] range.

  Args:
    documents: A sequence of document entries as a dictionary with the
      'children' key containing a sequence of bounding boxes.
    shuffle: random shuffle objects or not.

  Returns:
    The same sequence as `documents` with normalized coordinates according to
    the document width and height. Order is preserved.
  """
  normalized_documents = []

  def _normalize_entry(element, document_width, document_height):
    return {
        **element, "center": [
            _clamp(element["center"][0] / document_width),
            _clamp(element["center"][1] / document_height)
        ],
        "width": _clamp(element["width"] / document_width),
        "height": _clamp(element["height"] / document_height)
    }

  for document in documents:
    children = document["children"]
    document_width = float(document["width"])
    document_height = float(document["height"])
    normalize_fn = functools.partial(
        _normalize_entry,
        document_width=document_width,
        document_height=document_height)
    normalized_children = [normalize_fn(c) for c in children]
    if shuffle:
      random.Random(0).shuffle(normalized_children)
      # normalized_children = normalized_children
    else:
      normalized_children = sorted(
          normalized_children, key=lambda e: (e["center"][1], e["center"][0]))

    normalized_document = {**document, "children": normalized_children}
    normalized_documents.append(normalized_document)

  return normalized_documents


def get_dataset(batch_size,
                dataset_folder,
                n_devices,
                ds_file,
                max_length,
                add_bos=True,
                dataset_name="RICO",
                shuffle=False):
  """Obtain dataset from preprocessed json data.

  Args:
    batch_size: Number of samples in one batch.
    dataset_folder: Dataset folder.
    n_devices: Number of devices we use.
    ds_file: The data file name.
    max_length: the maximum length of input sequence.
    add_bos: Whether add bos and eos to the input sequence.
    dataset_name: The name of dataset.
    shuffle: Shuffle objects or not.

  Returns:
    One tf dataset.
    Vocab size in this dataset.
  """
  assert batch_size % n_devices == 0
  ds_path = os.path.join(dataset_folder, ds_file)
  # shuffle = True if "train" not in ds_file else shuffle
  dataset = LayoutDataset(dataset_name, ds_path, add_bos, shuffle)

  class_range = [dataset.offset_class, dataset.number_classes]
  center_x_range = [dataset.offset_center_x, dataset.resolution_w]
  center_y_range = [dataset.offset_center_y, dataset.resolution_h]
  width_range = [dataset.offset_width, dataset.resolution_w]
  height_range = [dataset.offset_height, dataset.resolution_h]
  pos_info = [
      class_range, width_range, height_range, center_x_range, center_y_range
  ]
  ds = dataset.setup_tf_dataset(
      batch_size, max_length, group_data_by_size=False)
  vocab_size = dataset.get_vocab_size()
  return ds, vocab_size, pos_info


def get_all_dataset(batch_size,
                    dataset_folder,
                    n_devices,
                    add_bos,
                    max_length,
                    dataset_name="RICO",
                    shuffle=False):
  """Creates datasets for various splits, such as train, valid and test.

  Args:
    batch_size: batch size of dataset loader.
    dataset_folder: path of dataset.
    n_devices: how many devices we will train our model on.
    add_bos: whether to add bos to the input sequence.
    max_length: the maximum length of input sequence.
    dataset_name: the name of the input dataset.
    shuffle: shuffle objects or not.
  Returns:
    datasets for various splits, the size of vocabulary and asset information.
  """
  train_ds, vocab_size, pos_info = get_dataset(batch_size, dataset_folder,
                                               n_devices, "train.json",
                                               max_length,
                                               add_bos,
                                               dataset_name,
                                               shuffle)
  eval_ds, _, _ = get_dataset(batch_size, dataset_folder, n_devices, "val.json",
                              max_length, add_bos, dataset_name, shuffle=False)
  test_ds, _, _ = get_dataset(batch_size, dataset_folder, n_devices,
                              "test.json", max_length, add_bos, dataset_name,
                              shuffle=False)
  return train_ds, eval_ds, test_ds, vocab_size, pos_info


class LayoutDataset:
  """Dataset for layout generation."""

  def __init__(self,
               dataset_name,
               path,
               add_bos = True,
               shuffle = False,
               resolution_w = DEFAULT_RESOLUTION_WIDTH,
               resolution_h = DEFAULT_RESOLUTION_HEIGHT,
               limit = 22):
    """Sets up the dataset instance, and computes the vocabulary.

    Args:
      dataset_name: The name of the input dataset, such as RICO.
      path: Path to the json file with the data. Raises ValueError if the data
        is faulty.
      add_bos: Whether add bos and eos to the input sequence.
      shuffle: shuffle objects or not.
      resolution_w: Discretization resolution to use for x and width
        coordinates.
      resolution_h: Discretization resolution to use for h and height
        coordinates.
      limit: Maximum amount of element in a layout.
    """
    with open(path, "r") as f:
      data = json.load(f)
    self.add_bos = add_bos
    self.dataset_name = datasets_info.DatasetName(dataset_name)
    self.data = _normalize_entries(data, shuffle)
    self.number_classes = datasets_info.get_number_classes(self.dataset_name)
    self.id_to_label = datasets_info.get_id_to_label_map(self.dataset_name)
    self.pad_idx, self.bos_idx, self.eos_idx = 0, 1, 2

    self.resolution_w = resolution_w
    self.resolution_h = resolution_h
    # ids of pad, bos and eos, unk are 0, 1, 2, 3, so we start from 4.
    self.offset_class = 4
    self.offset_center_x = self.offset_class + self.number_classes
    self.offset_center_y = self.offset_center_x + self.resolution_w

    self.offset_width = self.offset_center_y + self.resolution_h
    self.offset_height = self.offset_width + self.resolution_w
    self.limit = limit
    self.shuffle = shuffle

  def get_vocab_size(self):
    # Special symbols + num_classes +
    # all possible number of x, y, width and height positions.
    return self.offset_class + self.number_classes + (self.resolution_w +
                                                      self.resolution_h) * 2

  def _convert_entry_to_model_format(self, entries):
    """Converts a dataset entry to one sequence.

    E.g.:
    --> [BOS, entry1_pos, entry2_pos, ..., EOS]
    --> entry1_pos = class_id, center_x, center_y, width, height

    Args:
      entries: The sequence of bounding boxes to parse.

    Returns:
      One numpy array which contains positions of all items in the input entry.
      The first token and the last one is BOS and EOS symbols.
      Following previous works, we discrete the positon information according to
      resolution_w  and resolution_h.
    """
    processed_entry = []
    for box in entries[:self.limit]:
      category_id = box["category_id"]
      center = box["center"]
      width = box["width"]
      height = box["height"]
      class_id = category_id + self.offset_class
      discrete_x = round(center[0] *
                         (self.resolution_w - 1)) + self.offset_center_x
      discrete_y = round(center[1] *
                         (self.resolution_h - 1)) + self.offset_center_y
      # Clip the width and height of assets at least 1.
      discrete_width = round(
          np.clip(width * (self.resolution_w - 1), 1.,
                  self.resolution_w - 1)) + self.offset_width
      discrete_height = round(
          np.clip(height * (self.resolution_h - 1), 1.,
                  self.resolution_h - 1)) + self.offset_height
      processed_entry.extend(
          [class_id, discrete_width, discrete_height, discrete_x, discrete_y])
    if self.add_bos:
      # add bos and eos to the input seq
      processed_entry = [self.bos_idx] + processed_entry + [self.eos_idx]
    return np.array(processed_entry, dtype=np.int32)

  def boxes_iterator(self,):
    """Reads the dataset and produces an generator with each preprocessed entry.

    Yields:
      Preprocessed entry format for VTN model.
    """
    if self.shuffle:
      data = list(self.data)
      random.shuffle(data)
    else:
      data = self.data

    for entry in data:
      # if (max_number_elements is not None and
      #     len(entry["children"]) > max_number_elements):
      #   continue
      # # sorted_entry = sorted(
      # #     entry["children"], key=lambda e: (e["center"][1], e["center"][0]))
      # sorted_entry = entry["children"]
      # # random.shuffle(sorted_entry)
      # inputs = self._convert_entry_to_model_format(sorted_entry)
      inputs = self._convert_entry_to_model_format(entry["children"])
      yield inputs

  def setup_tf_dataset(
      self,
      batch_size,
      max_length,
      group_data_by_size):
    """Instantiates a tf.data.Dataset from a `dataset_parser`.

    Args:
      batch_size: The dataset instance is batched using this value.
      max_length: The maximum length of input sequence.
      group_data_by_size: If true, the data is batched grouping entries by their
        sequence length.

    Returns:
      An initialized dataset instance containing the same data as
      `dataset_parser`.
    """
    bucket_boundaries = (32, 52, 72, 102, 132, 152, 172, 192)
    dataset = tf.data.Dataset.from_generator(
        functools.partial(self.boxes_iterator),
        output_types=tf.int32, output_shapes=tf.TensorShape([None]))
    if group_data_by_size:
      dataset = dataset.apply(
          tf.data.experimental.bucket_by_sequence_length(
              element_length_func=lambda x: tf.shape(x)[0],
              bucket_boundaries=bucket_boundaries,
              bucket_batch_sizes=[
                  batch_size for _ in range(len(bucket_boundaries) + 1)
              ]))
    else:
      dataset = dataset.padded_batch(
          batch_size,
          padding_values=self.pad_idx,
          padded_shapes=max_length)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
