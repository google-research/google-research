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

"""Contains functions to load raw Tiny-ImageNet samples from their directory.
"""
import os
import matplotlib.image as mpimg
import numpy as np


NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 10000
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS


def get_annotations_map(val_annotations_path):
  val_annotations_file = open(val_annotations_path, 'r')
  val_annotations_contents = val_annotations_file.read()
  val_annotations = {}
  for line in val_annotations_contents.splitlines():
    pieces = line.strip().split()
    val_annotations[pieces[0]] = pieces[1]

  return val_annotations


def load_training_images(image_dir):
  """Loads training images and their labels."""
  image_index = 0

  images = np.ndarray(shape=(100000, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  names = []
  labels = []
  annotations = {}

  print('Loading training images from ', image_dir)
  # Loop through all the types directories
  c = 0
  for t in os.listdir(image_dir):
    annotations[t] = c
    if os.path.isdir(image_dir + t + '/images/'):
      type_images = os.listdir(image_dir + t + '/images/')
      # Loop through all the images of a type directory
      batch_index = 0
      # print ("Loading Class ", t)
      for image in type_images:
        image_file = os.path.join(image_dir, t + '/images/', image)
        image_data = mpimg.imread(image_file)
        if image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS):
          images[image_index, :] = image_data
          labels.append(c)
          names.append(image)

          image_index += 1
          batch_index += 1
    c += 1

  print('Loaded Training Images', image_index)
  return (images, np.asarray(labels), np.asarray(names)), annotations


def load_validation_images(testdir, annotations, batch_size=NUM_VAL_IMAGES):
  """Loads test images and their labels."""
  labels = []
  names = []
  image_index = 0
  images = np.ndarray(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  val_images = os.listdir(testdir + '/images/')

  # Loop through all the images of a val directory
  batch_index = 0

  val_annotations_map = get_annotations_map(testdir + 'val_annotations.txt')

  for image in val_images:
    image_file = os.path.join(testdir, 'images/', image)
    # reading the images as they are; no normalization, no color editing
    image_data = mpimg.imread(image_file)
    if image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS):
      images[image_index, :] = image_data
      image_index += 1
      labels.append(annotations[val_annotations_map[image]])
      names.append(image)
      batch_index += 1

    if batch_index >= batch_size:
      break

  print('Loaded Validation images ', image_index)
  return (images, np.asarray(labels), np.asarray(names))
