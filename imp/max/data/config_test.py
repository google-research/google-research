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

"""Tests for data_config."""

from absl.testing import absltest
from absl.testing import parameterized

from imp.max.core import constants
from imp.max.data import config as data_config
from imp.max.data import processing as proc_utils

TEST_DATA_ALL_MODALITIES = data_config.Dataset(
    name='test',
    data=data_config.ThirdPartyDataDefinition(
        tables={
            'train': 'a',
            'test': 'b'
        }, table='train'),
    modalities=data_config.Modalities(
        audio=data_config.Spectrogram(),
        vision=data_config.Vision(),
        text=data_config.Text(max_num_tokens=17),
    ))

TEST_DATA_ONLY_VIDEO = data_config.Dataset(
    name='test',
    data=data_config.ThirdPartyDataDefinition(
        tables={
            'train': 'a',
            'test': 'b'
        }, table='train'),
    modalities=data_config.Modalities(vision=data_config.Vision()))

TEST_DATA_ONLY_AUDIO = data_config.Dataset(
    name='test',
    data=data_config.ThirdPartyDataDefinition(
        tables={
            'train': 'a',
            'test': 'b'
        }, table='train'),
    modalities=data_config.Modalities(audio=data_config.Spectrogram()))

TEST_DATA_AUDIO_WITH_LABEL = data_config.Dataset(
    name='test',
    data=data_config.ThirdPartyDataDefinition(
        tables={
            'train': 'a',
            'test': 'b'
        }, table='train'),
    modalities=data_config.Modalities(audio=data_config.Spectrogram(
        annotation=data_config.Annotation(label=data_config.Label()),
        )))

MODIFIED_DATASET = data_config.Dataset(
    name='test',
    data=data_config.ThirdPartyDataDefinition(
        tables={
            'train': 'a',
            'test': 'b'
        }, table='test'),
    modalities=data_config.Modalities(
        audio=data_config.Spectrogram(num_test_clips=3),
        vision=data_config.Vision(multi_crop=True, min_resize=4),
        text=data_config.Text(max_num_tokens=17),
        )
    )


class DataTest(absltest.TestCase):

  def test_loader_from_modified_dataset(self):
    basic_loader = data_config.Loader(
        is_training=False,
        dataset=TEST_DATA_ALL_MODALITIES)
    copied = data_config.Loader(
        is_training=False,
        dataset=TEST_DATA_ALL_MODALITIES.copy_and_override({}))
    self.assertEqual(basic_loader, copied)
    copied_and_modified = data_config.Loader(
        dataset=TEST_DATA_ALL_MODALITIES.copy_and_override({
            'data': {
                'table': 'test'
            },
            'modalities': {
                'vision': {
                    'multi_crop': True,
                    'min_resize': 4
                },
                'audio': {
                    'num_test_clips': 3
                }
            }
        }),
        serving=constants.ServingStrategy.PRETRAIN,
        batch_size=16)
    self.assertNotEqual(basic_loader, copied_and_modified)
    self.assertEqual(copied_and_modified.dataset, MODIFIED_DATASET)
    self.assertEqual(copied_and_modified.serving,
                     constants.ServingStrategy.PRETRAIN)
    self.assertEqual(copied_and_modified.batch_size, 16)

  def test_experiment_data_post_init(self):
    experiment = data_config.ExperimentData(
        batch_size=8,
        vision_spatial_size=(110, 110),
        vision_spatial_patch_size=(11, 11),
        vision_temporal_size=4,
        vision_temporal_patch_size=7,
        waveform_temporal_size=30,
        waveform_temporal_patch_size=10,
        spectrogram_temporal_patch_size=2,
        spectrogram_spectoral_patch_size=16,
        text_size=9,
        is_training=None,
        shuffle=True,
        num_epochs=-1,
        loaders=(
            data_config.Loader(
                interval=1,
                is_training=False,
                dataset=TEST_DATA_ALL_MODALITIES.copy_and_override({})
                ),
            data_config.Loader(
                interval=2,
                is_training=False,
                dataset=TEST_DATA_ONLY_AUDIO,
                shuffle=False,
                ),
            data_config.Loader(
                interval=2,
                is_training=False,
                dataset=TEST_DATA_AUDIO_WITH_LABEL,
                shuffle=False,
                ),
            data_config.Loader(
                is_training=True,
                dataset=TEST_DATA_ONLY_VIDEO.copy_and_override({
                    'is_training': True}),
                batch_size=32
                ),
            ),
        )

    self.assertEqual(experiment.loaders[0].interval, 1)
    self.assertEqual(experiment.loaders[1].interval, 2)
    self.assertEqual(experiment.loaders[3].interval, 1)
    self.assertFalse(experiment.loaders[0].dataset.is_training)
    self.assertFalse(experiment.loaders[1].dataset.is_training)
    self.assertTrue(experiment.loaders[3].dataset.is_training)

    for i in range(3):
      self.assertTrue(experiment.loaders[i].shuffle)
      self.assertEqual(experiment.loaders[i].num_epochs, -1)
      self.assertEqual(experiment.loaders[i].batch_size, 8)

    expected_audio = data_config.Spectrogram(
        num_raw_waveform_samples=30, waveform_temporal_patch_size=10,
        temporal_patch_size=2, spectoral_patch_size=16)
    self.assertEqual(experiment.loaders[0].dataset.modalities.audio,
                     expected_audio)
    self.assertEqual(experiment.loaders[1].dataset.modalities.audio,
                     expected_audio)
    self.assertIsNone(experiment.loaders[3].dataset.modalities.audio)

    expected_audio_with_label = data_config.Spectrogram(
        num_raw_waveform_samples=30, waveform_temporal_patch_size=10,
        temporal_patch_size=2, spectoral_patch_size=16,
        annotation=data_config.Annotation(label=data_config.Label()))
    self.assertEqual(experiment.loaders[2].dataset.modalities.audio,
                     expected_audio_with_label)

    expected_vision = data_config.Vision(
        temporal_patch_size=7,
        spatial_patch_size=(11, 11),
        min_resize=110,
        crop_size=110,
        num_frames=4)
    self.assertEqual(experiment.loaders[0].dataset.modalities.vision,
                     expected_vision)
    expected_vision.min_resize = proc_utils.get_min_resize_value(110)
    self.assertEqual(experiment.loaders[3].dataset.modalities.vision,
                     expected_vision)
    self.assertIsNone(experiment.loaders[1].dataset.modalities.vision)

    expected_text = data_config.Text(max_num_tokens=9)
    self.assertEqual(experiment.loaders[0].dataset.modalities.text,
                     expected_text)
    self.assertIsNone(experiment.loaders[1].dataset.modalities.text)
    self.assertIsNone(experiment.loaders[3].dataset.modalities.text)


class MissConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'all_none',
          'batch_size': None,
          'is_training': None,
          'shuffle': None,
          'num_epochs': None,
      }, {
          'testcase_name': 'some_none',
          'batch_size': 8,
          'is_training': False,
          'shuffle': None,
          'num_epochs': -1,
      }, {
          'testcase_name': 'incorrect_num_epochs',
          'batch_size': 8,
          'is_training': False,
          'shuffle': True,
          'num_epochs': 0,
      }, {
          'testcase_name': 'incorrect_batch_size',
          'batch_size': 7,
          'is_training': False,
          'shuffle': True,
          'num_epochs': -1,
      })
  def test_experiment_data_invalid_config(
      self, batch_size, is_training, shuffle, num_epochs):
    with self.assertRaises(ValueError):
      data_config.ExperimentData(
          batch_size=batch_size,
          is_training=is_training,
          shuffle=shuffle,
          num_epochs=num_epochs,
          loaders=(
              data_config.Loader(
                  dataset=TEST_DATA_ALL_MODALITIES.copy_and_override({})
                  ),
              ),
          )

if __name__ == '__main__':
  absltest.main()
