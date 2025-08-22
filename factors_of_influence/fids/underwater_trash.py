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

"""Defines Underwater Trash dataset (ICRA'19) with object detection annotations.

URL: http://irvlab.cs.umn.edu/resources/trash-icra19
URL2: https://conservancy.umn.edu/handle/11299/214366

Paper:
Trash-icra19: A bounding box labeled dataset of underwater trash.
M. Fulton, J. Hong, and J. Sattar.
In Proc. Intl. Conf. on Robotics and Automation, 2020.
ArXiv: https://arxiv.org/pdf/1804.01079.pdf

The Underwater Trash Dataset (UTD) has been published at ICRA'19. The dataset is
sourced from the J-EDI dataset of marine debris. It contains images of many
different types of marine debris, captured from real-world environments,
providing a variety of objects in different states of decay, occlusion, and
overgrowth. Additionally, the clarity of the water and quality of the light
vary significantly from video to video. These videos were processed to extract
5,700 images, which comprise this dataset, all labeled with bounding boxes on
instances of trash, biological objects such as plants and animals, and ROVs.
"""
import os
from typing import List, Text
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from factors_of_influence import dataset_dirs
from factors_of_influence.fids import fids_dataset


UTD_DIR = dataset_dirs.UNDERWATER_TRASH_DIR

# The trash detection data has three classes:
# 1. Plastic: marine debris, all plastic materials.
# 2. ROV: All man-made objects(i.e., ROV, permanent sensors, etc),
#       intentionally placed in the environment.
# 3. Bio: All natural biological material, including fish, plants, and
#       biological detritus.
# Note: there are some more labels defined. We use the intersection over
#  train/val/test + background for all other annotations.
UTD_BOX_LABELS = ['background', 'metal', 'rov', 'bio', 'plastic']
UTD_BOX_LABELS_CONVERT = {
    'bio': 'bio',
    'cloth': 'background',
    'fishing': 'background',
    'metal': 'metal',
    'paper': 'background',
    'papper': 'background',
    'plastic': 'plastic',
    'platstic': 'plastic',
    'rov': 'rov',
    'rubber': 'background',
    'timestamp': 'background',
    'unknown': 'background',
    'wood': 'background',
}


def _label_name_to_one_hot(label_name):
  """Convert label name to one_hot vector."""
  label_name = UTD_BOX_LABELS_CONVERT[label_name]
  label_name_to_int = {n: i for i, n in enumerate(UTD_BOX_LABELS)}
  label_id = label_name_to_int[label_name]
  one_hot = np.zeros(shape=[len(UTD_BOX_LABELS)], dtype=np.float32)
  one_hot[label_id] = 1
  return one_hot


def _xml_find(xml, key):
  """Method to find in xml, which assert the value is not none."""
  # Required to keep pytype happy, given that xml.find returns Optional[...]
  key_element = xml.find(key)
  assert key_element is not None, f'Key {key} not found in xml {xml}'
  return key_element


def _xml_extract_img_size(xml):
  """Extract image size from XML."""
  size_xml = _xml_find(xml, 'size')
  return [
      float(_xml_find(size_xml, s).text) for s in ['width', 'height', 'depth']
  ]


def _extract_object_boxes_from_xml(xml):
  """Extract list of objects from XML."""
  box_list = []
  box_xmls = xml.findall('object')
  for box_xml in box_xmls:
    # Convert all XML fields (except bndbox)
    box_annotation = {x.tag: x.text for x in box_xml if x.tag != 'bndbox'}

    # Convert bndbox to box_shape
    box_shape_xml = _xml_find(box_xml, 'bndbox')
    box_shape = {
        key: int(_xml_find(box_shape_xml, key).text)
        for key in ['ymin', 'xmin', 'ymax', 'xmax']
    }
    box_annotation['bbox'] = box_shape

    box_list.append(box_annotation)

  return box_list


def _xml_extract_feature(xml_file, feature_name):
  """Extract features from XML file."""
  with tf.io.gfile.GFile(xml_file, mode='r') as f:
    xml_string = f.read()
  xml = ET.fromstring(xml_string)
  assert xml is not None, f'XML not parsed correctly from {xml_file}'
  xml_boxes = _extract_object_boxes_from_xml(xml)

  if feature_name == 'box_labels':
    return [_label_name_to_one_hot(box['name']) for box in xml_boxes]

  if feature_name == 'boxes':
    im_width, im_height, _ = _xml_extract_img_size(xml)

    box_list = []
    for box in xml_boxes:
      box_list.append(
          tfds.features.BBox(
              ymin=box['bbox']['ymin'] / im_height,
              xmin=box['bbox']['xmin'] / im_width,
              ymax=box['bbox']['ymax'] / im_height,
              xmax=box['bbox']['xmax'] / im_width,
          ))
    return box_list


class UnderwaterTrash(fids_dataset.FIDSDataset):
  """UnderwaterTrash dataset class."""

  def __init__(self):
    super().__init__(
        name='UTD',
        config_name='detection',
        feature_names=['image', 'boxes', 'box_labels'],
        splits=['train', 'validation', 'test'],
        feature_args={'box_labels': {'num_box_labels': len(UTD_BOX_LABELS)}},
    )

  def _info_features(self):
    return {'box_labels': UTD_BOX_LABELS}

  @staticmethod
  def _get_split_dir(split):
    """Return directory for split / test dir."""
    split_name = 'val' if split == 'validation' else split
    return f'{UTD_DIR}/dataset/{split_name}'

  def get_ids(self, split):
    # scrape ids from directory:
    split_pattern = f'{self._get_split_dir(split)}/*.jpg'
    img_files = tf.io.gfile.glob(split_pattern)
    ids_list = [
        os.path.splitext(os.path.basename(f))[0] for f in img_files
    ]
    return ids_list

  def get_feature(self, split, curr_id, feature_name):
    """Returns a feature. Can be a numpy array or path to an image."""
    split_dir = self._get_split_dir(split)

    if feature_name == 'image':
      img_file = f'{split_dir}/{curr_id}.jpg'
      return img_file, True

    if feature_name in ['boxes', 'box_labels']:
      xml_file = f'{split_dir}/{curr_id}.xml'
      return _xml_extract_feature(xml_file, feature_name), True

    raise ValueError(f'{feature_name} unknown')
