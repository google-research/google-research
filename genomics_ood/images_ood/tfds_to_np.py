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

r"""Convert TFDS to numpy arrays to make sure validation data split keeps the same.

Likelihood ratio method evaluates an input under both foreground and
background models. We need to make sure the inputs in the validation dataset
are always the smae set of inputs.

TFDS does not guarantee the split of the training and validation datasets
keeping the same. We convert TFDS to numpy arrays, save the data into disk, and
load the data from the disk for training foreground and background models.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

flags.DEFINE_string('out_dir', '/tmp/image_data', 'Directory to save datasets.')
flags.DEFINE_string(
    'name', 'fashion_mnist',
    'TFDS dataset name, fashion_mnist, mnist, cifar10, svhn_cropped')

FLAGS = flags.FLAGS

train_sample_size = {
    'fashion_mnist': 48000,
    'mnist': 48000,
    'cifar10': 40000,
    'svhn_cropped': 63257
}  # 20% of training is used for validation data


def tfds_to_np(dataset):
  """Convert tfds to numpy arrays."""
  images = []
  labels = []
  for example in tfds.as_numpy(dataset):
    image, label = example['image'], example['label']
    images.append(image)
    labels.append(label)

  images_np = np.vstack(images).reshape([-1] + list(image.shape))
  labels_np = np.vstack(labels).reshape(-1)

  return (images_np, labels_np)


def np_save(out_dir, name, subset, images_np, labels_np):
  print('dataset %s, subset %s, images.shape %s, labels.shape %s' %
        (name, subset, images_np.shape, labels_np.shape))
  with tf.compat.v1.gfile.Open(
      os.path.join(out_dir, '%s_%s.npy' % (name, subset)), 'wb') as f:
    np.save(f, images_np)
    np.save(f, labels_np)


def main(unused_argv):

  out_dir = FLAGS.out_dir
  tf.compat.v1.gfile.MakeDirs(out_dir)

  name = FLAGS.name

  data = tfds.load(name, as_dataset_kwargs={'shuffle_files': False})
  dataset = {}
  train_in0 = data['train']
  dataset['test'] = data['test']
  dataset['train'] = train_in0.take(train_sample_size[name])
  dataset['val'] = train_in0.skip(train_sample_size[name])

  for subset in ['train', 'val', 'test']:
    (images_np, labels_np) = tfds_to_np(dataset[subset])
    np_save(out_dir, name, subset, images_np, labels_np)


if __name__ == '__main__':
  app.run(main)
