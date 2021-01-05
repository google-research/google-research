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
"""Celeba preparation."""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

NP_PATH = os.path.join(os.environ['DATA_DIR'], 'celeba_npy')
ATTR_KEYS = frozenset([
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
    'Young'
])


def convert_dict_to_array(attr):
  """Converts attributes dictionary to array."""
  attr_array = np.zeros(len(ATTR_KEYS))
  for i, key in enumerate(ATTR_KEYS):
    if attr[key]:
      attr_array[i] = 1
  return attr_array


def get_numpy_array(ds, size=64):
  """Gets numpy array after crop."""
  image_array = np.stack([
      np.array(
          Image.fromarray(d['image'].numpy()[40:-20, 10:-10]).resize(
              (size, size))) for d in ds
  ])
  attr_array = np.stack([convert_dict_to_array(d['attributes']) for d in ds])
  return image_array, attr_array


def prepare_numpy_data(size=64):
  """Prepares numpy data."""
  builder = tfds.builder('celeb_a')
  try:
    builder.download_and_prepare()
  except KeyError:
    print('A version of tensorflow datasets might be incompatible.',
          'Consider installing tensorflow datasets version 4.0.0.')
  ds_train = builder.as_dataset(split='train')
  ds_test = builder.as_dataset(split='test')
  image_npy = f'image_{size}x{size}.npy'
  attr_npy = 'attr.npy'
  if tf.io.gfile.exists(os.path.join(NP_PATH, 'train', image_npy)):
    with tf.io.gfile.GFile(os.path.join(NP_PATH, 'train', image_npy),
                           'rb') as f_img, tf.io.gfile.GFile(
                               os.path.join(NP_PATH, 'train', attr_npy),
                               'rb') as f_attr:
      img_train = np.load(f_img)
      attr_train = np.load(f_attr)
      print('loaded from ' + os.path.join(NP_PATH, 'train', image_npy))
  else:
    img_train, attr_train = get_numpy_array(ds_train, size=size)
    if not tf.io.gfile.exists(os.path.join(NP_PATH, 'train')):
      tf.io.gfile.makedirs(os.path.join(NP_PATH, 'train'))
    with tf.io.gfile.GFile(os.path.join(NP_PATH, 'train', image_npy),
                           'wb') as f_img, tf.io.gfile.GFile(
                               os.path.join(NP_PATH, 'train', attr_npy),
                               'wb') as f_attr:
      np.save(f_img, img_train)
      np.save(f_attr, attr_train)
      print('saved at ' + os.path.join(NP_PATH, 'train', image_npy))
  if tf.io.gfile.exists(os.path.join(NP_PATH, 'test', image_npy)):
    with tf.io.gfile.GFile(os.path.join(NP_PATH, 'test', image_npy),
                           'rb') as f_img, tf.io.gfile.GFile(
                               os.path.join(NP_PATH, 'test', attr_npy),
                               'rb') as f_attr:
      img_test = np.load(f_img)
      attr_test = np.load(f_attr)
      print('loaded from ' + os.path.join(NP_PATH, 'test', image_npy))
  else:
    img_test, attr_test = get_numpy_array(ds_test, size=size)
    if not tf.io.gfile.exists(os.path.join(NP_PATH, 'test')):
      tf.io.gfile.makedirs(os.path.join(NP_PATH, 'test'))
    with tf.io.gfile.GFile(os.path.join(NP_PATH, 'test', image_npy),
                           'wb') as f_img, tf.io.gfile.GFile(
                               os.path.join(NP_PATH, 'test', attr_npy),
                               'wb') as f_attr:
      np.save(f_img, img_test)
      np.save(f_attr, attr_test)
      print('saved at ' + os.path.join(NP_PATH, 'test', image_npy))
  return (img_train, attr_train), (img_test, attr_test)


if __name__ == '__main__':
  prepare_numpy_data(size=64)
