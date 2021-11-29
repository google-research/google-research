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

from lingvo.core import spectrum_augmenter
import tensorflow as tf


class SpecAugment(tf.keras.layers.Layer):
  """A wrapper around lingo.core.spectrum_augmenter.SpectrumAugmenter .

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
  """

  def __init__(self,
               freq_mask_max_bins = 10,
               freq_mask_count = 2,
               time_mask_max_frames = 10,
               time_mask_count = 2,
               time_mask_max_ratio = 1.0,
               time_warp_max_frames = 8,
               time_warp_max_ratio = 1.0,
               use_input_dependent_random_seed = False):
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
        ops, with seeds determined by the input features (and timestamp).
    """
    super().__init__(name='SpecAugment')
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
    self.spec_augment_layer = spec_augment_params.Instantiate()

  def call(self, inputs, training=None):
    """Performs SpecAugment on the inputs.

    Args:
      inputs: input mel spectrogram of shape (batch_size, num_time_bins,
       num_freq_bins, channels).
      training: boolean training state.

    Returns:
      Augmented mel spectrogram of shape (batch_size, num_time_bins,
       num_freq_bins, channels).
    """
    if training is None:
      training = tf.keras.backend.learning_phase()
    if not training:
      return inputs

    batch_size = tf.shape(inputs)[0]
    num_time_bins = tf.shape(inputs)[1]
    paddings = tf.zeros((batch_size, num_time_bins))
    outputs = inputs
    outputs = self.spec_augment_layer._AugmentationNetwork(  # pylint: disable=protected-access
        inputs=outputs,
        paddings=paddings,
        global_seed=42)
    return outputs

