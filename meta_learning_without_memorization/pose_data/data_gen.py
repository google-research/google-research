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

"""Split images and labels into train and validation. Then pickle the data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
from absl import app
from absl import flags
from imageio import imread
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None,
                    'Root directory where images are stored.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data_folder = FLAGS.data_dir
  folders = [os.path.join(data_folder, family, character)  # pylint: disable=g-complex-comprehension
             for family in os.listdir(data_folder)
             if os.path.isdir(os.path.join(data_folder, family))
             for character in os.listdir(os.path.join(data_folder, family))]

  random.seed(1)
  random.shuffle(folders)

  metatrain_folders = folders[:-15]
  metaval_folders = folders[-15:]

  def normalize(labels_all):
    a1 = np.array(labels_all)
    x = a1[:, 1:]
    x_normed = (x - np.array([0, -1, 0])) / np.array([1, 2, 2 * np.pi])
    a1[:, 1:] = x_normed
    return a1.tolist()

  def get_images(path, shuffle=True):
    """Read images."""
    labels_and_images = []
    files = os.listdir(path)
    im_files = [f for f in files if f.endswith('png')]
    label_file = [f for f in files if 'label' in f]
    print(label_file)

    labels_all = pickle.load(open(os.path.join(path, label_file[0]), 'rb'))

    labels_all = normalize(labels_all)

    ims = im_files
    # first column in labels_all is id
    li = [(labels_all[int(os.path.splitext(im)[0])][1:],
           imread(os.path.join(path, im))) for im in ims]
    labels_and_images.extend(li)
    if shuffle:
      random.shuffle(labels_and_images)
    return labels_and_images

  #%%
  folders = metatrain_folders

  all_images = []
  all_labels = []

  for sampled_folders in folders:
    if os.listdir(sampled_folders):
      labels_and_images = get_images(sampled_folders, shuffle=False)

      labels = [li[0] for li in labels_and_images]
      images = [np.array(li[1]) for li in labels_and_images]
      all_images.append(images)
      all_labels.append(labels)

  with open('train_data.pkl', 'wb') as f:
    pickle.dump([all_images, all_labels], f)

  #%%
  folders = metaval_folders

  all_images = []
  all_labels = []

  for sampled_folders in folders:
    if os.listdir(sampled_folders):
      labels_and_images = get_images(sampled_folders, shuffle=False)

      labels = [li[0] for li in labels_and_images]
      images = [np.array(li[1]) for li in labels_and_images]
      all_images.append(images)
      all_labels.append(labels)

  with open('val_data.pkl', 'wb') as f:
    pickle.dump([all_images, all_labels], f)

if __name__ == '__main__':
  app.run(main)
