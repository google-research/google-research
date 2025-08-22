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

"""Data preprocessing for ImageNet2012 and CIFAR-10."""

from typing import Any, Callable

# pylint: disable=unused-import

from big_vision.pp import ops_general
from big_vision.pp import ops_image

# pylint: enable=unused-import

from big_vision.pp import utils
from big_vision.pp.builder import get_preprocess_fn as _get_preprocess_fn
from big_vision.pp.registry import Registry
import tensorflow as tf


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.247, 0.243, 0.261]


@Registry.register("preprocess_ops.random_crop_with_pad")
@utils.InKeyOutKey()
def get_random_crop_with_pad(crop_size,
                             padding):
  """Makes a random crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      random crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the random crop respectively.
    padding: how much to pad before cropping.

  Returns:
    A function, that applies random crop.
  """
  crop_size = utils.maybe_repeat(crop_size, 2)
  padding = utils.maybe_repeat(padding, 2)

  def _crop(image):
    image = tf.image.resize_with_crop_or_pad(image,
                                             crop_size[0] + padding[0],
                                             crop_size[1] + padding[1])
    return tf.image.random_crop(image,
                                [crop_size[0], crop_size[1], image.shape[-1]])

  return _crop


def preprocess_cifar(split, **_):
  """Preprocessing functions for CIFAR-10 training."""
  mean_str = ",".join([str(m) for m in CIFAR_MEAN])
  std_str = ",".join([str(m) for m in CIFAR_STD])
  if split == "train":
    pp = ("decode|"
          "value_range(0,1)|"
          "random_crop_with_pad(32,4)|"
          "flip_lr|"
          f"vgg_value_range(({mean_str}),({std_str}))|"
          "onehot(10, key='label', key_result='labels')|"
          "keep('image', 'labels')")
  else:
    pp = ("decode|"
          "value_range(0,1)|"
          "central_crop(32)|"
          f"vgg_value_range(({mean_str}),({std_str}))|"
          "onehot(10, key='label', key_result='labels')|"
          "keep('image', 'labels')")
  return _get_preprocess_fn(pp)


def preprocess_imagenet(split,
                        autoaugment = False,
                        label_smoothing = 0.0,
                        **_):
  """Preprocessing functions for ImageNet training."""
  if split == "train":
    pp = ("decode_jpeg_and_inception_crop(224)|"
          "flip_lr|")
    if autoaugment:
      pp += "randaug(2,10)|"
    pp += "value_range(-1,1)|"
    if label_smoothing:
      confidence = 1.0 - label_smoothing
      low_confidence = (1.0 - confidence) / (1000 - 1)
      pp += ("onehot(1000, key='label', key_result='labels', "
             f"on_value={confidence}, off_value={low_confidence})|")
    else:
      pp += "onehot(1000, key='label', key_result='labels')|"
    pp += "keep('image', 'labels')"
  else:
    pp = ("decode|"
          "resize_small(256)|"
          "central_crop(224)|"
          "value_range(-1,1)|"
          "onehot(1000, key='label', key_result='labels')|"
          "keep('image', 'labels')")
  return _get_preprocess_fn(pp)


PREPROCESS = {
    "cifar10": preprocess_cifar,
    "imagenet2012": preprocess_imagenet,
}


def get_preprocess_fn(dataset, split,
                      **preprocess_kwargs):
  """Makes a preprocessing function."""
  preprocess_fn_by_split = PREPROCESS.get(dataset, lambda _: (lambda x: x))
  split = "train" if "train" in split else "val"
  preprocess_fn = preprocess_fn_by_split(split, **preprocess_kwargs)
  return preprocess_fn

