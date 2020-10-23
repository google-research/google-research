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

"""Data loader for panorama stacks."""
import os
import pickle

import imageio
import numpy as np
from skimage.transform import resize
import tensorflow as tf


def load_sample_illuminations():
  """Loads example illuminations sampled from the test set.

  Returns:
    Numpy arrays of azimuth_npy and lc_npy from the data directory.
  """
  azimuth_npy = pickle.load(open("factorize_a_city/data/azimuth.npy", "rb"),
                            encoding="bytes")
  lc_npy = pickle.load(
      open("factorize_a_city/data/lighting_context.npy", "rb"),
      encoding="bytes")
  return azimuth_npy, lc_npy


def read_and_preprocess_panoramas(filepath):
  """Reads a panorama and applies the pole cropping preprocessing.

  Args:
    filepath: (str) The filepath to an image to read and process.

  Returns:
    Numpy array of panoramas of shape [384, 960, 3] that takes values from
    [0, 1]
  """
  im = imageio.imread(filepath)[:, :, :3] / 255.
  # Resize to 480, 960.
  im = resize(im, [480, 960])

  # Crop the bottom 20% of the panorama which consist mostly of roads.
  im = im[:-96]
  # [0, 1] ranged panorama
  return im


def read_stack(filepath, require_alignment=True):
  """Reads a stack of panorama and applies the pole cropping preprocessing.

  Inside filepath should be a set of panoramas of the format:
    <fileapth>/00.png
    <filepath>/01.png
    ...
    <filepath>/N.png

  If require_alignment is true, will load the stack alignment parameters from:
    <filepath>/alignment.npy

  Args:
    filepath: (str) The filepath to a directory of panoramas to load.
    require_alignment: (bool) If true, loads alignment.npy.

  Returns:
    A numpy array of panoramas of shape [S, 384, 960, 3], where S is the number
    of panorama images in filepath. If require_alignment is True, also returns
    alignment parameters of shape [S, 8, 32, 2].
  """
  num_files = len(tf.io.gfile.glob(os.path.join(filepath, "*.png")))
  ims = []
  for i in range(num_files):
    ims.append(
        read_and_preprocess_panoramas(os.path.join(filepath, "%02d.png" % i)))
  stacked_ims = np.stack(ims, axis=0)
  if require_alignment:
    alignment_fp = os.path.join(filepath, "alignment.npy")
    alignment = pickle.load(open(alignment_fp, "rb"), encoding="bytes")

    if alignment.shape[0] != stacked_ims.shape[0]:
      raise ValueError("Mis-matched number of images and alignment parameters")
    return stacked_ims, alignment
  return stacked_ims


def load_learned_warp_parameters(filepath):
  """Loads the pickle file of alignment parameters at filepath.

  Args:
    filepath: (str) A filepath to an alignment.npy file.

  Returns:
    Numpy alignment parameters of shape [S, H, W, 2] for aligning stacks using
    libs.image_alignment where S is the number of panoramas in a stack.
  """
  return np.load(filepath)


def write_stack_images(filepath, stack_tensor, prefix=""):
  """Writes stack_tensor Numpy images to filepath.

  Each batch index of stack_tensor corresponds to an image to save. The images
  are saved in
  <filepath>/<batch index>.png
  or if prefix is provided,
  <filepath>/<prefix>_<batch index>.png

  Args:
    filepath: (str) A filepath to a directory to save images.
    stack_tensor: [S, H, W, 3] a Numpy array of S images to write ranging from
      [0, 1].
    prefix: (str) A string to prefix the file names.
  """
  stack_tensor = np.clip(255. * stack_tensor, 0, 255).astype(np.uint8)
  if not os.path.exists(filepath):
    os.makedirs(filepath)
  for i in range(stack_tensor.shape[0]):
    if prefix:
      output_fp = os.path.join(filepath, "%s_%02d.png" % (prefix, i))
    else:
      output_fp = os.path.join(filepath, "%02d.png" % i)
    imageio.imsave(output_fp, stack_tensor[i])
