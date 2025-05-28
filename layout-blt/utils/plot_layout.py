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

"""Plot layout."""
from typing import Sequence, Tuple, Union

import cv2
from gvt.projects.layout_gvt.datasets import coco_info
from gvt.projects.layout_gvt.datasets import magazine_info
from gvt.projects.layout_gvt.datasets import publaynet_info
from gvt.projects.layout_gvt.datasets import rico_info
import numpy as np
import tensorflow as tf


def parse_entry(
    dataset, entry
):
  """Parses a dataset entry according to its dataset.

  Args:
    dataset: Name of the dataset type.
    entry: [asset_dim] (class_id, width, height, center_x, center_y). Entry in
      the layoutvae network output format.

  Returns:
    A tuple with the class id, the class name, an associated color, and the
      bounding box.
  """
  if dataset == "RICO":
    info = rico_info
  elif dataset == "PubLayNet":
    info = publaynet_info
  elif dataset == "MAGAZINE":
    info = magazine_info
  elif dataset == "COCO":
    info = coco_info
  else:
    raise ValueError(f"Dataset '{dataset}' not found")
  class_id = entry[0]
  class_name = info.ID_TO_LABEL[class_id]
  color = info.COLORS[class_name]
  bounding_box = entry[1:]

  return class_id, class_name, color, bounding_box


def parse_layout_sample(data, dataset_type):
  """Decode to a sequence of bounding boxes."""
  result = {}
  for idx in range(0, data.shape[-1], 5):
    entry = data[idx:idx+5]
    _, class_name, _, bounding_box = parse_entry(dataset_type, entry)

    width, height, center_x, center_y = bounding_box
    # Adds a small number to make sure .5 can be rounded to 1.
    x_min = np.round(center_x - width / 2. + 1e-4)
    x_max = np.round(center_x + width / 2. + 1e-4)
    y_min = np.round(center_y - height / 2. + 1e-4)
    y_max = np.round(center_y + height / 2. + 1e-4)

    x_min = np.clip(x_min / 31., 0., 1.)
    y_min = np.clip(y_min / 31., 0., 1.)
    x_max = np.clip(x_max / 31., 0., 1.)
    y_max = np.clip(y_max / 31., 0., 1.)
    result[class_name] = [np.clip(bounding_box / 31., 0., 1.),
                          [y_min, x_min, y_max, x_max]]
  return result


def plot_sample(data,
                target_width,
                target_height,
                dataset_type,
                border_size = 1,
                thickness = 4):
  """Draws an image from a sequence of bounding boxes.

  Args:
    data: A sequence of bounding boxes. They must be in the 'networks output'
      format (see dataset_entries_to_network_outputs).
    target_width: Result image width.
    target_height: Result image height.
    dataset_type: Dataset type keyword. Necessary to assign labels.
    border_size: Width of the border added to the image.
    thickness: It is the thickness of the rectangle border line in px.
      Thickness of -1 px will display each box with a colored box without text.

  Returns:
    The image as an np.ndarray of np.uint8 type.
  """
  image = np.zeros((target_height, target_width, 3), dtype=np.uint8) + 255

  for idx in range(0, data.shape[-1], 5):
    entry = data[idx:idx+5]
    _, class_name, color, bounding_box = parse_entry(dataset_type, entry)

    width, height, center_x, center_y = bounding_box
    # Adds a small number to make sure .5 can be rounded to 1.
    x_min = np.round(center_x - width / 2. + 1e-4)
    x_max = np.round(center_x + width / 2. + 1e-4)
    y_min = np.round(center_y - height / 2. + 1e-4)
    y_max = np.round(center_y + height / 2. + 1e-4)

    x_min = round(np.clip(x_min / 31., 0., 1.) * target_width)
    y_min = round(np.clip(y_min / 31., 0., 1.) * target_height)
    x_max = round(np.clip(x_max / 31., 0., 1.) * target_width)
    y_max = round(np.clip(y_max / 31., 0., 1.) * target_height)

    image = cv2.rectangle(
        image,
        pt1=(x_min, y_min),
        pt2=(x_max, y_max),
        color=color,
        thickness=thickness)
    textsize = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]

    # get coords based on boundary
    textx = (x_max + x_min - textsize[0]) / 2
    texty = (y_min + y_max + textsize[1]) / 2
    # if thickness != -1:
    image = cv2.putText(
        image,
        text=class_name,
        org=(int(textx), int(texty)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.,
        color=(0, 0, 0),
        thickness=2)

  image = cv2.copyMakeBorder(
      image,
      top=border_size,
      bottom=border_size,
      left=border_size,
      right=border_size,
      borderType=cv2.BORDER_CONSTANT,
      value=[0, 0, 0])

  return image
