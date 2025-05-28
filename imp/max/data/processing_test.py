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

"""Tests for processing."""

from absl.testing import absltest
import chex
import tensorflow as tf

from imp.max.core import constants
from imp.max.data import processing
from imp.max.data.datasets import prompts

_N_DEVICES = 2
DataFeatureName = constants.DataFeatureName


def setUpModule():
  chex.set_n_cpu_devices(_N_DEVICES)


class ProcessingTest(tf.test.TestCase):

  def test_get_shape(self):
    fixed = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    fixed_shape = processing.get_shape(fixed)
    self.assertListEqual(fixed_shape, [2, 3])
    dynamic, _ = tf.unique(tf.convert_to_tensor([1, 2, 3, 4, 5, 6]))
    dynamic_shape = processing.get_shape(dynamic)
    self.assertListEqual(dynamic_shape, [6])

  def test_remove_key(self):
    features_dict = {'a': 1, 'b': 2, 'c': 3}
    output = processing.remove_key(features_dict, 'b')
    self.assertDictEqual(output, {'a': 1, 'c': 3})
    with self.assertRaises(KeyError):
      processing.remove_key(features_dict, 'b')

  def test_extend_waveform_dim(self):
    input_tensor = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]],
                                        dtype=tf.float32)
    output1 = processing.extend_waveform_dim(input_tensor)
    self.assertAllEqual(output1, tf.reshape(input_tensor, (2, 3, 1)))
    output3 = processing.extend_waveform_dim(input_tensor, 3)
    self.assertAllEqual(output3, tf.reshape(input_tensor, (3, 2, 1)))

  def test_label_smoothing(self):
    input_tensor = tf.convert_to_tensor([1, 2, 3, 4, 5], dtype=tf.float32)
    unsmoothed = processing.label_smoothing(input_tensor, alpha=0)
    expected = input_tensor
    self.assertAllClose(unsmoothed, expected)
    smoothed = processing.label_smoothing(input_tensor)
    self.assertAllClose(smoothed, expected, atol=0.5)
    self.assertNotAllClose(smoothed, expected)
    fully_smoothed = processing.label_smoothing(input_tensor, alpha=1)
    class_weight = 0.2
    expected_fully_smoothed = tf.convert_to_tensor([class_weight] * 5)
    self.assertAllClose(fully_smoothed, expected_fully_smoothed)

  def test_batched_mixup(self):
    features = tf.convert_to_tensor(
        [[[1, 1], [2, 2], [3, 3]], [[2, 3], [3, 4], [4, 5]],
         [[3, 5], [4, 6], [5, 7]], [[4, 7], [5, 8], [6, 9]]],
        dtype=tf.float32)
    labels = tf.convert_to_tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]],
                                  dtype=tf.float32)
    features_dict = {
        DataFeatureName.TOKEN_RAW: features,
        DataFeatureName.LABEL: labels,
    }
    tf.random.set_seed(0)
    mixed = processing.batched_mixup(
        features_dict, feature_name=DataFeatureName.TOKEN_RAW,
        label_feature_name=DataFeatureName.LABEL)
    expected_features = tf.convert_to_tensor([[[1.1169643, 1.2339287],
                                               [2.1169643, 2.2339287],
                                               [3.1169643, 3.2339287]],
                                              [[2.6039233, 4.2078466],
                                               [3.6039233, 5.2078466],
                                               [4.6039233, 6.2078466]],
                                              [[2.764873, 4.5297456],
                                               [3.7648728, 5.529746],
                                               [4.7648726, 6.5297456]],
                                              [[3.716111, 6.432222],
                                               [4.716111, 7.432222],
                                               [5.7161107, 8.432221]]])
    expected_labels = tf.convert_to_tensor([[1.116964, 2.116964, 3.116964],
                                            [2.905885, 3.905885, 4.905885],
                                            [2.764873, 3.764873, 4.764873],
                                            [4.432222, 5.432221, 6.432222]])
    self.assertDictEqual(
        mixed, {
            DataFeatureName.TOKEN_RAW: expected_features,
            DataFeatureName.LABEL: expected_labels
        })

    del features_dict[DataFeatureName.LABEL]
    with self.assertRaises(KeyError):
      processing.batched_mixup(
          features_dict, feature_name=DataFeatureName.TOKEN_RAW,
          label_feature_name=DataFeatureName.LABEL)

  def test_random_crop_resize(self):
    num_frames = 5
    num_channels = 3
    frames = tf.random.uniform((num_frames, 2, 4, num_channels))
    cropped = processing.random_crop_resize(
        frames,
        output_h=1,
        output_w=2,
        aspect_ratio=(1, 1),
        area_range=(1., 1.))
    self.assertAllEqual(cropped.shape.as_list(),
                        (num_frames, 1, 2, num_channels))
    cropped_large = processing.random_crop_resize(
        frames,
        output_h=4,
        output_w=8,
        aspect_ratio=(1, 1),
        area_range=(1., 1.))
    self.assertAllEqual(cropped_large.shape.as_list(),
                        (num_frames, 4, 8, num_channels))

  def test_multi_crop_image(self):
    num_frames = 5
    num_channels = 3
    frames = tf.random.uniform((num_frames, 2, 4, num_channels))
    cropped = processing.multi_crop_image(
        frames, target_height=1, target_width=2)
    self.assertAllEqual(cropped.shape.as_list(),
                        (3 * num_frames, 1, 2, num_channels))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      processing.multi_crop_image(frames, target_height=3, target_width=3)

  def test_label_to_sentence(self):
    label = tf.convert_to_tensor(['cat', 'dog'])
    with self.assertRaises(NotImplementedError):
      processing.label_to_sentence(label, 'unknown')
    image_labels = processing.label_to_sentence(
        label,
        modality='image',
        eval_image_promtps=prompts.BASE_IMAGE_PROMPTS)
    expected_image_labels = tf.convert_to_tensor([
        'This is a photo of cat and dog.', 'This photo shows cat and dog.',
        'You can see cat and dog in this photo.',
        'There is a depiction of cat and dog in this photo.',
        'This image contains cat and dog.'
    ])
    self.assertAllEqual(image_labels, expected_image_labels)
    video_labels = processing.label_to_sentence(
        label,
        modality='video',
        eval_video_promtps=prompts.BASE_VIDEO_PROMPTS)
    expected_video_labels = tf.convert_to_tensor([
        'This is a video of cat and dog.', 'This video clip shows cat and dog.',
        'You can see cat and dog in this video.',
        'There is a depiction of cat and dog over time in this video.',
        'This video clip contains cat and dog.'
    ])
    self.assertAllEqual(video_labels, expected_video_labels)
    audio_labels = processing.label_to_sentence(
        label,
        modality='audio',
        eval_audio_promtps=prompts.BASE_AUDIO_PROMPTS)
    # pylint: disable=line-too-long
    expected_audio_labels = tf.convert_to_tensor([
        'This is an audio recording of cat and dog.',
        'This audio clip shows cat and dog.',
        'You hear the sound of cat and dog in this audio clip.',
        'A sound of cat and dog can be heard over time in this audio recording.',
        'This audio clip contains cat and dog.'
    ])
    # pylint: enable=line-too-long
    self.assertAllEqual(audio_labels, expected_audio_labels)

  def test_label_to_sentence_max_length(self):
    label = tf.convert_to_tensor(['cat', 'dog'])
    image_labels = processing.label_to_sentence(
        label,
        modality='image',
        max_num_sentences=3,
        eval_image_promtps=prompts.BASE_IMAGE_PROMPTS)
    expected_image_labels = tf.convert_to_tensor([
        'This is a photo of cat and dog.',
        'This photo shows cat and dog.',
        'You can see cat and dog in this photo.',
    ])
    self.assertAllEqual(image_labels, expected_image_labels)


if __name__ == '__main__':
  absltest.main()
