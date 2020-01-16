# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Regularizer classes for semi-supervised image classification."""
import abc
import math

import jax
import jax.numpy as jnp

from masking import box_mask
from masking import cow_mask


class Regularizer(object):
  """Abstract base regularizer."""

  @abc.abstractmethod
  def perturb_sample(self, image_batch, rng_key):
    pass

  @abc.abstractmethod
  def mix_images(self, image0_batch, image1_batch, rng_key):
    pass

  @abc.abstractmethod
  def mix_samples(self, image0_batch, y0_batch, image1_batch, y1_batch,
                  rng_key):
    image_batch, blend_factors = self.mix_images(
        image0_batch, image1_batch, rng_key)
    y_batch = y0_batch + (y1_batch - y0_batch) * blend_factors[:, None]
    return image_batch, y_batch, blend_factors


class IdentityRegularizer(object):
  """Identify regularizer.

  Perturb is no-op.
  """

  def perturb_sample(self, image_batch, rng_key):  # pylint: disable=unused-argument
    return image_batch

  @abc.abstractmethod
  def mix_images(self, image0_batch, image1_batch, rng_key):
    pass


class ICTRegularizer(Regularizer):
  """Interpolation Consistency Training regularizer."""

  def __init__(self, alpha=0.1):
    self.alpha = alpha

  def mix_images(self, image0_batch, image1_batch, rng_key):
    n_samples = len(image0_batch)
    blend_factors = jax.random.beta(
        rng_key, self.alpha, self.alpha, (n_samples,))
    image_batch = image0_batch + (image1_batch - image0_batch) * \
        blend_factors[:, None, None, None]
    return image_batch, blend_factors


class BoxMaskRegularizer(Regularizer):
  """Box mask regularizer."""

  def __init__(self, backg_noise_std, mask_prob, scale_mode, scale,
               random_aspect_ratio):
    if scale_mode == 'fixed':
      self.scale = scale
    elif scale_mode in {'random_size', 'random_area'}:
      self.scale = scale_mode
    else:
      raise ValueError('Unknown scale_mode \'{}\''.format(scale_mode))
    self.backg_noise_std = backg_noise_std
    self.mask_prob = mask_prob
    self.random_aspect_ratio = random_aspect_ratio

  def perturb_sample(self, image_batch, rng_key):
    mask_size = image_batch.shape[1:3]
    prob_key, box_key, noise_key = jax.random.split(rng_key, num=3)
    boxes = box_mask.generate_boxes(
        N=len(image_batch), mask_size=mask_size, scale=self.scale,
        random_aspect_ratio=self.random_aspect_ratio, rng=box_key)
    masks = box_mask.box_masks(boxes, mask_size)
    if self.mask_prob < 1.0:
      b = jax.random.bernoulli(prob_key, self.mask_prob,
                               shape=(len(image_batch), 1, 1, 1))
      b = b.astype(jnp.float32)
      masks = 1.0 + (masks - 1.0) * b
    if self.backg_noise_std > 0.0:
      noise = jax.random.normal(noise_key, image_batch.shape) * \
          self.backg_noise_std
      return image_batch * masks + noise * (1.0 - masks)
    else:
      return image_batch * masks

  def mix_images(self, image0_batch, image1_batch, rng_key):
    n_samples = len(image0_batch)
    mask_size = image0_batch.shape[1:3]
    boxes = box_mask.generate_boxes(
        N=n_samples, mask_size=mask_size, scale=self.scale,
        random_aspect_ratio=self.random_aspect_ratio, rng=rng_key)
    masks = box_mask.box_masks(boxes, mask_size)
    blend_factors = masks.mean(axis=(1, 2, 3))
    image_batch = image0_batch + (image1_batch - image0_batch) * masks
    return image_batch, blend_factors


class CowMaskRegularizer(Regularizer):
  """CowMask regularizer."""

  def __init__(self, backg_noise_std, mask_prob, cow_sigma_range,
               cow_prop_range):
    self.backg_noise_std = backg_noise_std
    self.mask_prob = mask_prob
    self.cow_sigma_range = cow_sigma_range
    self.cow_prop_range = cow_prop_range
    self.log_sigma_range = (math.log(cow_sigma_range[0]),
                            math.log(cow_sigma_range[1]))
    self.max_sigma = cow_sigma_range[1]

  def perturb_sample(self, image_batch, rng_key):
    mask_size = image_batch.shape[1:3]
    prob_key, cow_key, noise_key = jax.random.split(rng_key, num=3)
    masks = cow_mask.cow_masks(
        len(image_batch), mask_size, self.log_sigma_range, self.max_sigma,
        self.cow_prop_range, cow_key)
    if self.mask_prob < 1.0:
      b = jax.random.bernoulli(prob_key, self.mask_prob,
                               shape=(len(image_batch), 1, 1, 1))
      b = b.astype(jnp.float32)
      masks = 1.0 + (masks - 1.0) * b
    if self.backg_noise_std > 0.0:
      noise = jax.random.normal(noise_key, image_batch.shape) * \
          self.backg_noise_std
      return image_batch * masks + noise * (1.0 - masks)
    else:
      return image_batch * masks

  def mix_images(self, image0_batch, image1_batch, rng_key):
    n_samples = len(image0_batch)
    mask_size = image0_batch.shape[1:3]
    masks = cow_mask.cow_masks(
        n_samples, mask_size, self.log_sigma_range, self.max_sigma,
        self.cow_prop_range, rng_key)
    blend_factors = masks.mean(axis=(1, 2, 3))
    image_batch = image0_batch + (image1_batch - image0_batch) * masks
    return image_batch, blend_factors
