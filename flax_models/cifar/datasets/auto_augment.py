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
r"""Functions used in the AutoAugment paper.

AutoAugment: Learning Augmentation Policies from Data
https://arxiv.org/abs/1805.09501

Contrary to
https://github.com/tensorflow/models/tree/master/research/autoaugment,
we don't rely on PIL. As a result, the augmentation function can be converted to
a graph by Tensorflow, which allows us to use `num_parallel_calls` when mapping
it to the dataset and greatly reduce the time needed to preprocess the data.

It has been checked that this implementation is equivalent to the one referenced
above.
"""

from typing import Callable, Dict, List, Tuple
import tensorflow as tf

from flax_models.cifar.datasets import autoaugment_utils


# Shorthand notation for augmentation function typing.
# The function takes as input two tensors (image and strength of the
# augmentation), and returns a tensor (the augmented image).
AugmentationFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


_PADDING_CONSTANT = [125, 123, 114]  # Mean value per channel for cifar.


@tf.function
def _shear_x(image, strength):
  """Shears the image horizontally.

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the augmentation. Should be between 0 (no shearing)
      and 0.3 (maximum shearing).

  Returns:
    The sheared image.
  """
  if tf.random.uniform([1]) > 0.5:
    strength = -strength
  return autoaugment_utils.shear_x(
      image, strength, replace=tf.constant(_PADDING_CONSTANT, dtype=tf.uint8))


@tf.function
def _shear_y(image, strength):
  """Shears the image vertically.

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the augmentation. Should be between 0 (no shearing)
      and 0.3 (maximum shearing).

  Returns:
    The sheared image.
  """
  if tf.random.uniform([1]) > 0.5:
    strength = -strength
  return autoaugment_utils.shear_y(
      image, strength, replace=tf.constant(_PADDING_CONSTANT, dtype=tf.uint8))


@tf.function
def _translate_x_abs(image, strength):
  """Translates the image horizontally.

  The image will be padded with _PADDING_CONSTANT.

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the augmentation. How many pixels should the image
      being translated by. Should be between 0 and 10 for cifar (although higher
      values are possible, they were not considered in the original paper).

  Returns:
    The translated image.
  """
  if tf.random.uniform([1]) > 0.5:
    strength = -strength
  return autoaugment_utils.translate_x(
      image, strength, replace=tf.constant(_PADDING_CONSTANT, dtype=tf.uint8))


@tf.function
def _translate_y_abs(image, strength):
  """Translates the image vertically.

  The image will be padded with _PADDING_CONSTANT.

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the augmentation. How many pixels should the image
      being translated by. Should be between 0 and 10 for cifar (although higher
      values are possible, they were not considered in the original paper).

  Returns:
    The translated image.
  """
  if tf.random.uniform([1]) > 0.5:
    strength = -strength
  return autoaugment_utils.translate_y(
      image, strength, replace=tf.constant(_PADDING_CONSTANT, dtype=tf.uint8))


@tf.function
def _rotate(image, strength):
  """Rotates the image.

  The image will be padded with _PADDING_CONSTANT.

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the augmentation. How many degrees should the image
      being rotated by. Should be between 0 and 30 for cifar (although higher
      values are possible, they were not considered in the original paper).

  Returns:
    The rotated image.
  """
  if tf.random.uniform([1]) > 0.5:
    strength = -strength
  strength = tf.cast(strength, tf.float32)
  return autoaugment_utils.rotate(
      image, strength, replace=tf.constant(_PADDING_CONSTANT, dtype=tf.uint8))


@tf.function
def _auto_contrast(image, _):
  """Applies auto-contrast to the image (as defined in PIL).

  Args:
    image: A tf.uint8 tensor.

  Returns:
    The contrasted image.
  """
  return autoaugment_utils.autocontrast(image)


@tf.function
def _invert(image, _):
  """Inverts the colors in the image (returns the "negative").

  Args:
    image: A tf.uint8 tensor.

  Returns:
    The inverted image.
  """
  return tf.ones_like(image) * 255 - image


@tf.function
def _equalize(image, _):
  """Equalizes the image (as defined in PIL).

  Args:
    image: A tf.uint8 tensor.

  Returns:
    The equalized image.
  """
  return autoaugment_utils.equalize(image)


@tf.function
def _flip(image, _):
  """Flips the image horizontally.

  Args:
    image: A tf.uint8 tensor.

  Returns:
    The flipped image.
  """
  return tf.reverse(image, tf.constant([0, 1]))


@tf.function
def _solarize(image, strength):
  """Solarizes the image (as defined in PIL).

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the solarization. Should be between 255 (no
      solarization) and 0 (maximum solarization).

  Returns:
    The solarized image.
  """
  strength = tf.cast(tf.math.floor(strength), tf.uint8)
  return autoaugment_utils.solarize(image, strength)


@tf.function
def _posterize(image, strength):
  """Posterizes the image (as defined in PIL).

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the posterization. Should be between 4 (less
      posterization) and 0 (maximum posterization).

  Returns:
    The solarized image.
  """
  strength = tf.cast(tf.math.ceil(strength), tf.uint8)
  return autoaugment_utils.posterize(image, strength)


@tf.function
def _contrast(image, strength):
  """Contrasts the image (as defined in PIL).

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the contrast. Should be between 0.1
      (the contrast of the image is strongly reduced) and 1.9 (the contrast of
      the image is strongly increased). strength=1 does nothing.

  Returns:
    The contrasted image.
  """
  degenerate = tf.image.rgb_to_grayscale(image)
  mean = tf.reduce_mean(tf.cast(degenerate, tf.float32))
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return autoaugment_utils.blend(degenerate, image, strength)


@tf.function
def _color(image, strength):
  """Adjusts the color of the image (as defined in PIL).

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the adjustment. Should be between 0.1 (the color of
      the image is strongly reduced) and 1.9 (the color of the image is strongly
      increased). strength=1 does nothing.

  Returns:
    The colorized image.
  """
  return autoaugment_utils.color(image, strength)


@tf.function
def _brightness(image, strength):
  """Adjusts the brightness of the image (as defined in PIL).

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the adjustment. Should be between 0.1 (the brightness
      of the image is strongly reduced) and 1.9 (the brightness of the image is
      strongly increased). strength=1 does nothing.

  Returns:
    The image with adjusted brightness.
  """
  return autoaugment_utils.brightness(image, strength)


@tf.function
def _sharpness(image, strength):
  """Adjusts the sharpness of the image (as defined in PIL).

  Args:
    image: A tf.uint8 tensor.
    strength: Strength of the adjustment. Should be between 0.1 (the sharpness
      of the image is strongly reduced) and 1.9 (the sharpness of the image is
      strongly increased). strength=1 does nothing.

  Returns:
    The image with adjusted sharpness.
  """
  return autoaugment_utils.sharpness(image, strength)


@tf.function
def _cutout_abs(image, strength):
  """Randomly cutout patches from the image.

  Args:
    image: A tf.uint8 tensor.
    strength: Length (in pixels) of the patch to cutout. Should be
      between 0 and 20 for cifar (although higher values are possible, they were
      not considered in the original paper).

  Returns:
    The image with a patch cut out.
  """
  if strength < 0:
    return image
  strength = tf.cast(tf.math.ceil(strength), tf.int32)
  replace = tf.constant(_PADDING_CONSTANT, dtype=tf.uint8)
  return autoaugment_utils.cutout(image, strength // 2, replace=replace)


def _maybe_apply_function(func, image,
                          level, prob):
  """Randomly applies `func` to the image with probability `prob`.

  Args:
    func: The augmentation function to maybe apply to the image
    image: A uint8 tensor, the image to augment.
    level: Strength of the augmentation. Should be a value compatible with the
      given augmentation function.
    prob: Probability with which the augmentation should be applied.

  Returns:
    The (possibly augmented) image.
  """
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob),
      tf.bool)
  augmented_image = tf.cond(should_apply_op, lambda: func(image, level),
                            lambda: image)
  return augmented_image


def _select_and_apply_random_policy(policies,
                                    image):
  """Randomly selects an augmentation policy and applies it to the image.

  Taken from third_party/cloud_tpu/models/efficientnet/autoaugment.py.

  Args:
    policies: A list of possible policies to sample from.
    image: A uint8 tensor.

  Returns:
    The image augmented with a random policy.
  """
  policy_to_select = tf.random.uniform([],
                                       maxval=len(policies),
                                       dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image),
        lambda: image)
  return image


def _available_augmentations(
):
  """Returns a name to function mapping for the augmentations.

  To each policy name is associated a tuple containing the augmentation function
  and the acceptable range for its strength.

  Returns:
    A dictionary mapping the name of an available transformation to a tuple
    (operation, min_strength, max_strength) where operation is a function that
    takes an image and a strength (magnitude of the transformation) and returns
    a transformed version of the image. Each operation has a minimum strength
    and a maximum strength associated to it (NB: As per in the original paper,
    minimum stength does not always mean that the image is the less affected,
    see the `_contrast` augmentation for instance).
  """
  l = {
      'ShearX': (_shear_x, 0, 0.3),
      'ShearY': (_shear_y, 0, 0.3),
      'Cutout': (_cutout_abs, 0, 20),
      'TranslateX': (_translate_x_abs, 0, 10),
      'TranslateY': (_translate_y_abs, 0, 10),
      'Rotate': (_rotate, 0, 30),
      'AutoContrast': (_auto_contrast, 0, 1),
      'Invert': (_invert, 0, 1),
      'Equalize': (_equalize, 0, 1),
      'Solarize': (_solarize, 255, 0),
      'Posterize': (_posterize, 4, 0),
      'Contrast': (_contrast, 0.1, 1.9),
      'Color': (_color, 0.1, 1.9),
      'Brightness': (_brightness, 0.1, 1.9),
      'Sharpness': (_sharpness, 0.1, 1.9),
  }
  return l


def _get_good_policies_cifar():
  """Returns the AutoAugment policies found on Cifar.

  A policy is composed of two augmentations applied sequentially to the image.
  Each augmentation is described as a tuple where the first element is the
  type of transformation to apply, the second is the probability with which the
  augmentation should be applied, and the third element is the strength of the
  transformation.
  """
  exp0_0 = [
      [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
      [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
      [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
      [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)]]
  exp0_1 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
      [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
      [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
      [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)]]
  exp0_2 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.0, 2)],
      [('TranslateY', 0.7, 9), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.9, 0), ('Solarize', 0.4, 3)],
      [('Equalize', 0.7, 5), ('Invert', 0.1, 3)],
      [('TranslateY', 0.7, 9), ('TranslateY', 0.7, 9)]]
  exp0_3 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 1)],
      [('TranslateY', 0.8, 9), ('TranslateY', 0.9, 9)],
      [('AutoContrast', 0.8, 0), ('TranslateY', 0.7, 9)],
      [('TranslateY', 0.2, 7), ('Color', 0.9, 6)],
      [('Equalize', 0.7, 6), ('Color', 0.4, 9)]]
  exp1_0 = [
      [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
      [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
      [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
      [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
      [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)]]
  exp1_1 = [
      [('Brightness', 0.3, 7), ('AutoContrast', 0.5, 8)],
      [('AutoContrast', 0.9, 4), ('AutoContrast', 0.5, 6)],
      [('Solarize', 0.3, 5), ('Equalize', 0.6, 5)],
      [('TranslateY', 0.2, 4), ('Sharpness', 0.3, 3)],
      [('Brightness', 0.0, 8), ('Color', 0.8, 8)]]
  exp1_2 = [
      [('Solarize', 0.2, 6), ('Color', 0.8, 6)],
      [('Solarize', 0.2, 6), ('AutoContrast', 0.8, 1)],
      [('Solarize', 0.4, 1), ('Equalize', 0.6, 5)],
      [('Brightness', 0.0, 0), ('Solarize', 0.5, 2)],
      [('AutoContrast', 0.9, 5), ('Brightness', 0.5, 3)]]
  exp1_3 = [
      [('Contrast', 0.7, 5), ('Brightness', 0.0, 2)],
      [('Solarize', 0.2, 8), ('Solarize', 0.1, 5)],
      [('Contrast', 0.5, 1), ('TranslateY', 0.2, 9)],
      [('AutoContrast', 0.6, 5), ('TranslateY', 0.0, 9)],
      [('AutoContrast', 0.9, 4), ('Equalize', 0.8, 4)]]
  exp1_4 = [
      [('Brightness', 0.0, 7), ('Equalize', 0.4, 7)],
      [('Solarize', 0.2, 5), ('Equalize', 0.7, 5)],
      [('Equalize', 0.6, 8), ('Color', 0.6, 2)],
      [('Color', 0.3, 7), ('Color', 0.2, 4)],
      [('AutoContrast', 0.5, 2), ('Solarize', 0.7, 2)]]
  exp1_5 = [
      [('AutoContrast', 0.2, 0), ('Equalize', 0.1, 0)],
      [('ShearY', 0.6, 5), ('Equalize', 0.6, 5)],
      [('Brightness', 0.9, 3), ('AutoContrast', 0.4, 1)],
      [('Equalize', 0.8, 8), ('Equalize', 0.7, 7)],
      [('Equalize', 0.7, 7), ('Solarize', 0.5, 0)]]
  exp1_6 = [
      [('Equalize', 0.8, 4), ('TranslateY', 0.8, 9)],
      [('TranslateY', 0.8, 9), ('TranslateY', 0.6, 9)],
      [('TranslateY', 0.9, 0), ('TranslateY', 0.5, 9)],
      [('AutoContrast', 0.5, 3), ('Solarize', 0.3, 4)],
      [('Solarize', 0.5, 3), ('Equalize', 0.4, 4)]]
  exp2_0 = [
      [('Color', 0.7, 7), ('TranslateX', 0.5, 8)],
      [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
      [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],
      [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
      [('Solarize', 0.5, 2), ('Invert', 0.0, 3)]]
  exp2_1 = [
      [('AutoContrast', 0.1, 5), ('Brightness', 0.0, 0)],
      [('Cutout', 0.2, 4), ('Equalize', 0.1, 1)],
      [('Equalize', 0.7, 7), ('AutoContrast', 0.6, 4)],
      [('Color', 0.1, 8), ('ShearY', 0.2, 3)],
      [('ShearY', 0.4, 2), ('Rotate', 0.7, 0)]]
  exp2_2 = [
      [('ShearY', 0.1, 3), ('AutoContrast', 0.9, 5)],
      [('TranslateY', 0.3, 6), ('Cutout', 0.3, 3)],
      [('Equalize', 0.5, 0), ('Solarize', 0.6, 6)],
      [('AutoContrast', 0.3, 5), ('Rotate', 0.2, 7)],
      [('Equalize', 0.8, 2), ('Invert', 0.4, 0)]]
  exp2_3 = [
      [('Equalize', 0.9, 5), ('Color', 0.7, 0)],
      [('Equalize', 0.1, 1), ('ShearY', 0.1, 3)],
      [('AutoContrast', 0.7, 3), ('Equalize', 0.7, 0)],
      [('Brightness', 0.5, 1), ('Contrast', 0.1, 7)],
      [('Contrast', 0.1, 4), ('Solarize', 0.6, 5)]]
  exp2_4 = [
      [('Solarize', 0.2, 3), ('ShearX', 0.0, 0)],
      [('TranslateX', 0.3, 0), ('TranslateX', 0.6, 0)],
      [('Equalize', 0.5, 9), ('TranslateY', 0.6, 7)],
      [('ShearX', 0.1, 0), ('Sharpness', 0.5, 1)],
      [('Equalize', 0.8, 6), ('Invert', 0.3, 6)]]
  exp2_5 = [
      [('AutoContrast', 0.3, 9), ('Cutout', 0.5, 3)],
      [('ShearX', 0.4, 4), ('AutoContrast', 0.9, 2)],
      [('ShearX', 0.0, 3), ('Posterize', 0.0, 3)],
      [('Solarize', 0.4, 3), ('Color', 0.2, 4)],
      [('Equalize', 0.1, 4), ('Equalize', 0.7, 6)]]
  exp2_6 = [
      [('Equalize', 0.3, 8), ('AutoContrast', 0.4, 3)],
      [('Solarize', 0.6, 4), ('AutoContrast', 0.7, 6)],
      [('AutoContrast', 0.2, 9), ('Brightness', 0.4, 8)],
      [('Equalize', 0.1, 0), ('Equalize', 0.0, 6)],
      [('Equalize', 0.8, 4), ('Equalize', 0.0, 4)]]
  exp2_7 = [
      [('Equalize', 0.5, 5), ('AutoContrast', 0.1, 2)],
      [('Solarize', 0.5, 5), ('AutoContrast', 0.9, 5)],
      [('AutoContrast', 0.6, 1), ('AutoContrast', 0.7, 8)],
      [('Equalize', 0.2, 0), ('AutoContrast', 0.1, 2)],
      [('Equalize', 0.6, 9), ('Equalize', 0.4, 4)]]
  exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
  exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
  exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
  return  exp0s + exp1s + exp2s


def _get_good_policies_svhn():
  """Returns the AutoAugment policies found on SVHN.

  A policy is composed of two augmentations applied sequentially to the image.
  Each augmentation is described as a tuple where the first element is the
  type of transformation to apply, the second is the probability with which the
  augmentation should be applied, and the third element is the strength of the
  transformation.
  """
  return [[('ShearX', 0.9, 4), ('Invert', 0.2, 3)],
          [('ShearY', 0.9, 8), ('Invert', 0.7, 5)],
          [('Equalize', 0.6, 5), ('Solarize', 0.6, 6)],
          [('Invert', 0.9, 3), ('Equalize', 0.6, 3)],
          [('Equalize', 0.6, 1), ('Rotate', 0.9, 3)],
          [('ShearX', 0.9, 4), ('AutoContrast', 0.8, 3)],
          [('ShearY', 0.9, 8), ('Invert', 0.4, 5)],
          [('ShearY', 0.9, 5), ('Solarize', 0.2, 6)],
          [('Invert', 0.9, 6), ('AutoContrast', 0.8, 1)],
          [('Equalize', 0.6, 3), ('Rotate', 0.9, 3)],
          [('ShearX', 0.9, 4), ('Solarize', 0.3, 3)],
          [('ShearY', 0.8, 8), ('Invert', 0.7, 4)],
          [('Equalize', 0.9, 5), ('TranslateY', 0.6, 6)],
          [('Invert', 0.9, 4), ('Equalize', 0.6, 7)],
          [('Contrast', 0.3, 3), ('Rotate', 0.8, 4)],
          [('Invert', 0.8, 5), ('TranslateY', 0.0, 2)],
          [('ShearY', 0.7, 6), ('Solarize', 0.4, 8)],
          [('Invert', 0.6, 4), ('Rotate', 0.8, 4)],
          [('ShearY', 0.3, 7), ('TranslateX', 0.9, 3)],
          [('ShearX', 0.1, 6), ('Invert', 0.6, 5)],
          [('Solarize', 0.7, 2), ('TranslateY', 0.6, 7)],
          [('ShearY', 0.8, 4), ('Invert', 0.8, 8)],
          [('ShearX', 0.7, 9), ('TranslateY', 0.8, 3)],
          [('ShearY', 0.8, 5), ('AutoContrast', 0.7, 3)],
          [('ShearX', 0.7, 2), ('Invert', 0.1, 5)]]


def get_autoaugment_fn(dataset_name):
  """Returns the optimal AutoAugmentation function for a given dataset.

  The returned function takes as input an image (uint8 tensor) and returns its
  augmented version. This function can be mapped on Tensorflow Datasets.

  Args:
    dataset_name: Name of the dataset for which we should return the optimal
      policy.
  """
  if 'cifar' in dataset_name:
    auto_aug_policies = _get_good_policies_cifar()
  elif 'svhn' in dataset_name:
    auto_aug_policies = _get_good_policies_svhn()
  else:
    raise ValueError('AutoAugment policies only available for cifar and svhn.')

  aug_list = _available_augmentations()

  def _augmentation_to_fn(
      augmentation):
    """Takes and augmentation tuple and returns the associated function.

    Args:
      augmentation: A tuple (name, probability, magnitude) where name is the
        name of the transformation (refer to _available_augmentations to see
        which are available), probability is the probability with which the
        transformation should be applied to the image, and magnitude is the
        strength of the transformation, as an integer between 0 and 10. In the
        original paper, the strength of the operation is given as an integer
        between 0 and 10 (included), to have a common scale for policy
        searching. We thus need to scale this strength to fit in the
        (min_strength, max_strength) range of the operation.

    Returns:
      An image augmentation function that takes as input a single image and
        returns an augmented version.
    """
    name, prob, magnitude = augmentation
    magnitude = magnitude / 10
    f_aug, min_strength, max_strength = aug_list[name]

    def f(image):
      level = min_strength + magnitude * (max_strength - min_strength)
      return _maybe_apply_function(
          f_aug, image, tf.constant(level), tf.constant(prob))

    return f

  def _parse_policy(
      policy):
    """Returns the augmentation function associated to a given policy.

    Args:
      policy: A list of two augmentation tuples (see `_augmentation_to_fn` for
        more details).
    """
    augmentation_1, augmentation_2 = policy
    f1 = _augmentation_to_fn(augmentation_1)
    f2 = _augmentation_to_fn(augmentation_2)
    return lambda x: f2(f1(x))

  auto_augment_policies = [_parse_policy(p) for p in auto_aug_policies]

  def _augment_image(image):
    return _select_and_apply_random_policy(auto_augment_policies, image)

  return _augment_image
