# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import os
import numpy as np
import tensorflow as tf
# pylint: skip-file

ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
          'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
          'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
          'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
          'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
          'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
          'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
          'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
          'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
          'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
          'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
          'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
          'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
ID_ATT = {v: k for k, v in ATT_ID.items()}

CENTRAL_FRACTION = 0.89
LOAD_SIZE = 142 #286
CROP_SIZE = 128 #256

def load_train(image_path, label):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [LOAD_SIZE, LOAD_SIZE])
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_crop(image, [CROP_SIZE, CROP_SIZE, 3])
  image = tf.clip_by_value(image, 0, 255) / 127.5 - 1
  label = (label + 1) // 2
  image = tf.cast(image, tf.float32)
  label = tf.cast(label, tf.int32)
  return (image, label)

def load_test(image_path, label):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [LOAD_SIZE, LOAD_SIZE])
  image = tf.image.central_crop(image, CENTRAL_FRACTION)
  image = tf.clip_by_value(image, 0, 255) / 127.5 - 1
  label = (label + 1) // 2
  image = tf.cast(image, tf.float32)
  label = tf.cast(label, tf.int32)
  return (image, label)


# load entire training dataset
def data_train(image_path, label_path, batch_size):
  img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
  img_paths = np.array([os.path.join(image_path, img_name) for img_name in img_names])
  labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
  labels = labels[:,ATT_ID['Male']]
  n_examples = img_names.shape[0]

  train_dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
  train_dataset = train_dataset.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.shuffle(batch_size*128)
  # train_dataset = train_dataset.shuffle(n_examples)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.repeat().prefetch(1)

  train_iter = train_dataset.make_one_shot_iterator()
  batch = train_iter.get_next()

  return batch, int(np.ceil(n_examples/batch_size))

def data_test(image_path,label_path, batch_size):
  img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
  img_paths = np.array([os.path.join(image_path, img_name) for img_name in img_names])
  labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
  labels = labels[:,ATT_ID['Male']]
  n_examples = img_names.shape[0]

  test_dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
  test_dataset = test_dataset.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  test_dataset = test_dataset.repeat().prefetch(1)

  test_iter = test_dataset.make_one_shot_iterator()
  batch = test_iter.get_next()

  return batch, int(np.ceil(n_examples/batch_size))





