# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 162770
CENTRAL_FRACTION = 0.89
LOAD_SIZE = 142 #286
CROP_SIZE = 128 #256

def cal_eo(a, y_label, y_pred):
  a = np.array(a)
  y_label = np.array(y_label)
  y_pred = np.array(y_pred)

  idx00 = np.logical_and(a==0,y_label==0)
  idx01 = np.logical_and(a==0,y_label==1)
  idx10 = np.logical_and(a==1,y_label==0)
  idx11 = np.logical_and(a==1,y_label==1)

  d00 = 1 - np.sum(y_pred[idx00])/y_pred[idx00].shape[0]
  d01 = np.sum(y_pred[idx01])/y_pred[idx01].shape[0]
  d10 = 1 - np.sum(y_pred[idx10])/y_pred[idx10].shape[0]
  d11 = np.sum(y_pred[idx11])/y_pred[idx11].shape[0]

  eo = np.abs(d00-d10)+np.abs(d01-d11)
  return (d00,d01,d10,d11,eo)

def reorg(label_path,af,bf):
  img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
  labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
  entry = np.concatenate((img_names[:, np.newaxis], labels), axis=1)
  a = np.asarray((labels[:,ATT_ID[af]]+1)//2)
  b = np.asarray((labels[:,ATT_ID[bf]]+1)//2)
  d00 = []
  d01 = []
  d10 = []
  d11 = []
  for i in range(labels.shape[0]):
      if a[i]==0:
          if b[i]==0: d00.append(entry[i])
          elif b[i]==1: d01.append(entry[i])
      elif a[i]==1:
          if b[i]==0: d10.append(entry[i])
          elif b[i]==1: d11.append(entry[i])
  min_leng = np.min([len(d00),len(d01),len(d10),len(d11)])
  new_list = d00[:min_leng]+d01[:3*min_leng]+d10[:3*min_leng]+d11[:min_leng]
  return np.array(new_list)

def reorg_fake(label_path,af,bf):
  img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
  labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
  entry = np.concatenate((img_names[:, np.newaxis], labels), axis=1)
  a = np.asarray((labels[:,ATT_ID[af]]+1)//2)
  b = np.asarray((labels[:,ATT_ID[bf]]+1)//2)
  d00 = []
  d01 = []
  d10 = []
  d11 = []
  for i in range(labels.shape[0]):
      if a[i]==0:
          if b[i]==0: d00.append(entry[i])
          elif b[i]==1: d01.append(entry[i])
      elif a[i]==1:
          if b[i]==0: d10.append(entry[i])
          elif b[i]==1: d11.append(entry[i])
  min_leng = np.min([len(d00),len(d01),len(d10),len(d11)])
  new_list = d00[:min_leng]+d01[:3*min_leng]+d10[:3*min_leng]+d11[:min_leng]
  return np.array(new_list)

def load_train(image_path, label, att):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [LOAD_SIZE, LOAD_SIZE])
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_crop(image, [CROP_SIZE, CROP_SIZE, 3])
  image = tf.clip_by_value(image, 0, 255) / 127.5 - 1
  label = (label + 1) // 2
  att = (att + 1) // 2
  image = tf.cast(image, tf.float32)
  label = tf.cast(label, tf.int32)
  att = tf.cast(att, tf.int32)
  return (image, label, att)

def load_test(image_path, label, att):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [LOAD_SIZE, LOAD_SIZE])
  image = tf.image.central_crop(image, CENTRAL_FRACTION)
  image = tf.clip_by_value(image, 0, 255) / 127.5 - 1
  label = (label + 1) // 2
  att = (att + 1) // 2
  image = tf.cast(image, tf.float32)
  label = tf.cast(label, tf.int32)
  att = tf.cast(att, tf.int32)
  return (image, label, att)

# load balanced training dataset
def data_train(image_path, label_path, batch_size):
  a = 'Male'
  b = 'Arched_Eyebrows'
  new_entry = reorg(label_path,a,b)
  n_examples = new_entry.shape[0]
  img_names = new_entry[:,0]
  img_paths = np.array([os.path.join(image_path, img_name) for img_name in img_names])
  img_labels = new_entry[:,1:]
  labels = img_labels[:,ATT_ID['Arched_Eyebrows']].astype(int)
  att = img_labels[:,ATT_ID['Male']].astype(int)

  train_dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels, att))
  train_dataset = train_dataset.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.shuffle(n_examples)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.repeat().prefetch(1)

  train_iter = train_dataset.make_one_shot_iterator()
  batch = train_iter.get_next()

  return batch, int(np.ceil(n_examples/batch_size))

def data_fake(image_path, label_path, batch_size):
  a = 'Male'
  b = 'Arched_Eyebrows'
  new_entry = reorg_fake(label_path,a,b)
  n_examples = new_entry.shape[0]
  img_names = new_entry[:,0]
  img_paths = np.array([os.path.join(image_path, img_name) for img_name in img_names])
  img_labels = new_entry[:,1:]
  labels = img_labels[:,ATT_ID['Arched_Eyebrows']].astype(int)
  att = img_labels[:,ATT_ID['Male']].astype(int)

  train_dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels, att))
  train_dataset = train_dataset.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.shuffle(n_examples)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.repeat().prefetch(1)

  train_iter = train_dataset.make_one_shot_iterator()
  batch = train_iter.get_next()

  return batch, int(np.ceil(n_examples/batch_size))

# load entire training dataset
# def data_train(image_path, label_path, batch_size):
#     img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
#     img_paths = np.array([os.path.join(image_path, img_name) for img_name in img_names])
#     labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
#     n_examples = img_names.shape[0]
#     # labels = labels[:,ATT_ID['Male']]
#     labels = labels[:,ATT_ID['Smiling']]
#     # labels = labels[:,ATT_ID['Arched_Eyebrows']]


#     train_dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
#     train_dataset = train_dataset.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     train_dataset = train_dataset.shuffle(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, seed=0)
#     train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
#     train_dataset = train_dataset.repeat().prefetch(1)

#     train_iter = train_dataset.make_one_shot_iterator()
#     batch = train_iter.get_next()

#     return batch, int(np.ceil(n_examples/batch_size))

def data_test(image_path, label_path, batch_size):
  img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
  img_paths = np.array([os.path.join(image_path, img_name) for img_name in img_names])
  img_labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
  n_examples = img_names.shape[0]
  labels = img_labels[:,ATT_ID['Arched_Eyebrows']]
  att = img_labels[:,ATT_ID['Male']]

  test_dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels, att))
  test_dataset = test_dataset.map(load_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  test_dataset = test_dataset.repeat().prefetch(1)

  test_iter = test_dataset.make_one_shot_iterator()
  batch = test_iter.get_next()

  return batch, int(np.ceil(n_examples/batch_size))




