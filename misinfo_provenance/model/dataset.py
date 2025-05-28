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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file implements mixup and cutmix data augmentations."""
import tensorflow.compat.v2 as tf
from tensorflow_addons import image as tfa_image

tfa_cutout = tfa_image.cutout
tfa_blur = tfa_image.gaussian_filter2d
tfa_rotate = tfa_image.rotate
layers = tf.keras.layers
IMG_SIZE = 256
"""
Cut Mix Augmentation.
"""


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
  gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
  gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
  return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


@tf.function
def get_box(lambda_value):
  """Generate Bounding box.

  Args:
      lambda_value:

  Returns:
  """
  cut_rat = tf.math.sqrt(1.0 - lambda_value)

  cut_w = IMG_SIZE * cut_rat  # rw
  cut_w = tf.cast(cut_w, tf.int32)

  cut_h = IMG_SIZE * cut_rat  # rh
  cut_h = tf.cast(cut_h, tf.int32)

  cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE,
                            dtype=tf.int32)  # rx
  cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE,
                            dtype=tf.int32)  # ry

  boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
  boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
  bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
  bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

  target_h = bby2 - boundaryy1
  if target_h == 0:
    target_h += 1

  target_w = bbx2 - boundaryx1
  if target_w == 0:
    target_w += 1

  return boundaryx1, boundaryy1, target_h, target_w


@tf.function
def cutmix(image1, image2):
  """Cutmix.

  Args:
    image1:
    image2:

  Returns:

  """
  # (image1, label1), (image2, label2) = train_ds_one, train_ds_two.

  # Same parameters used in SSCD.
  alpha = [2]
  beta = [0.25]

  # Get a sample from the Beta distribution.
  lambda_value = sample_beta_distribution(1, alpha, beta)

  # Define Lambda.
  lambda_value = lambda_value[0][0]

  # Get the bounding box offsets, heights and widths.
  boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

  # Get a patch from the second image (`image2`).
  crop2 = tf.image.crop_to_bounding_box(image2, boundaryy1, boundaryx1,
                                        target_h, target_w)
  # Pad the `image2` patch (`crop2`) with the same offset.
  image2 = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE,
                                        IMG_SIZE)
  # Get a patch from the first image (`image1`).
  crop1 = tf.image.crop_to_bounding_box(image1, boundaryy1, boundaryx1,
                                        target_h, target_w)
  # Pad the `image1` patch (`crop1`) with the same offset.
  img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE,
                                      IMG_SIZE)

  # Modify the first image by subtracting the patch from `image1`
  # (before applying the `image2` patch).
  image1 = image1 - img1
  # Add the modified `image1` and `image2`  together to get the CutMix image.
  image = image1 + image2

  # Adjust Lambda in accordance to the pixel ration.
  lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
  lambda_value = tf.cast(lambda_value, tf.float32)

  # Combine the labels of both images.
  return image  #, label


def mix_up(images_one, images_two, alpha=0.8):
  """Mixup.

  Args:
    images_one:
    images_two:
    alpha:

  Returns:

  """

  batch_size = tf.shape(images_one)[0]

  # Sample lambda and reshape it to do the mixup.
  l = sample_beta_distribution(batch_size, alpha, alpha)
  x_l = tf.reshape(l, (batch_size, 1, 1, 1))
  # y_l = tf.reshape(l, (batch_size, 1))

  # Perform mixup on both images and labels by combining a pair of images/labels
  # (one from each dataset) into one image/label.
  images = images_one * x_l + images_two * (1 - x_l)
  return images
