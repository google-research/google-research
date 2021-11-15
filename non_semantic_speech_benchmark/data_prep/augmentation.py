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

"""Augmentation library for spec augment for keras transform.
"""

import abc
import enum

from lingvo.core import spectrum_augmenter
import tensorflow as tf  # We want TF2.


class AugmentationMode(enum.Enum):
  """The supported alignment modes."""
  DISABLED = 'disabled'
  TRAIN_ONLY = 'train_only'
  TEST_ONLY = 'test_only'
  TRAIN_AND_TEST = 'train_and_test'


class Augmentation(tf.keras.Model, abc.ABC):
  """Abstract base class for augmentation."""

  def __init__(self,
               augment_mode = AugmentationMode.TRAIN_ONLY):
    """Builds Augmentation.

    Args:
      augment_mode: the augmentation mode.
    """
    super().__init__()
    self.augment_mode = augment_mode

  def _should_augment(self, training = False):
    return (training and self.augment_mode in [
        AugmentationMode.TRAIN_ONLY, AugmentationMode.TRAIN_AND_TEST
    ]) or (not training and self.augment_mode in [
        AugmentationMode.TEST_ONLY, AugmentationMode.TRAIN_AND_TEST
    ])

  def call(self, inputs, training = False):
    if self._should_augment(training):
      return self.apply_augmentation(inputs)
    else:
      return inputs

  @abc.abstractmethod
  def apply_augmentation(self, inputs):
    pass


class SpecAugment(Augmentation):
  """A wrapper around lingo.core.spectrum_augmenter.SpectrumAugmenter.

   SpecAugment is a data augmentation that combines three transformations:
   - a time warping of up to max(time_warp_max_frames,
   time_warp_max_ratio*input_length) frames.
   - a masking of sampled frequencies with zeros along the entire time axis
   (freq_mask)
   - a masking of sampled timesteps with zeros along the entire frequency axis
   (time_mask)
   For the frequency mask, freq_mask_max_bins is the maximum number of
   consecutive frequency bins to be masked, freq_mask_count is the number of
   masks to apply to a signal. Same for time_mask.

   Note: SpecAugment takes mel spectrograms as input.
  """

  def __init__(self,
               freq_mask_max_bins,
               freq_mask_count,
               time_mask_max_frames,
               time_mask_count,
               time_mask_max_ratio,
               time_warp_max_frames,
               time_warp_max_ratio,
               use_input_dependent_random_seed = True,
               augment_mode = AugmentationMode.TRAIN_ONLY):
    """Builds SpecAugment.

    Args:
      freq_mask_max_bins: max number of consecutive mel bins to mask in a band.
      freq_mask_count: number of frequency bands to mask.
      time_mask_max_frames: max number of consecutive time frames to mask.
      time_mask_count: number of time bands to mask.
      time_mask_max_ratio: max time mask ratio.
      time_warp_max_frames: max numer of time frames to warp.
      time_warp_max_ratio: max ratio of the time warp.
      use_input_dependent_random_seed: If true, uses stateless random TensorFlow
        ops, with seeds determined by the input features.
      augment_mode: the augmentation mode.
    """
    super().__init__(augment_mode)
    spec_augment_params = spectrum_augmenter.SpectrumAugmenter.Params()
    spec_augment_params.freq_mask_max_bins = freq_mask_max_bins
    spec_augment_params.freq_mask_count = freq_mask_count
    spec_augment_params.time_mask_max_frames = time_mask_max_frames
    spec_augment_params.time_mask_count = time_mask_count
    spec_augment_params.time_warp_max_frames = time_warp_max_frames
    spec_augment_params.time_warp_max_ratio = time_warp_max_ratio
    spec_augment_params.time_mask_max_ratio = time_mask_max_ratio
    spec_augment_params.use_input_dependent_random_seed = (
        use_input_dependent_random_seed)
    spec_augment_params.name = 'SpecAugmentLayer'
    self._spec_augment_layer = spec_augment_params.Instantiate()

  def apply_augmentation(self, inputs):
    """Performs SpecAugment on the inputs.

    Args:
      inputs: input mel spectrogram of shape (num_time_bins, num_freq_bins) or
        (batch_size, num_time_bins, num_freq_bins).

    Returns:
      Augmented mel spectrogram of shape (num_time_bins, num_freq_bins) or
        (batch_size, num_time_bins, num_freq_bins).
    """
    if inputs.shape.ndims == 2:
      inputs = inputs[None, :, :, None]
      squeeze_axis = [0, 3]
    elif inputs.shape.ndims == 3:
      inputs = inputs[:, :, :, None]
      squeeze_axis = 3
    else:
      raise ValueError('Input shape must have 2 or 3 dimensions')

    outputs, _ = self._spec_augment_layer.FPropDefaultTheta(
        inputs=inputs,
        paddings=tf.zeros(tf.shape(inputs)[:2])
        )
    return tf.squeeze(outputs, axis=squeeze_axis)

