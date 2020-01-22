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

"""Tests model train, based on tensorflow/examples/speech_commands."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from kws_streaming.train import model_flags
from kws_streaming.train import train


FLAGS = flags.FLAGS


# Used to convert a dictionary into an object, for mocking parsed flags.
class DictStruct(object):

  def __init__(self, **entries):
    self.__dict__.update(entries)


class TrainTest(tf.test.TestCase, parameterized.TestCase):

  def _GetWavData(self):
    with self.cached_session():
      sample_data = tf.zeros([32000, 2])
      wav_encoder = tf.audio.encode_wav(sample_data, 16000)
      wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _SaveTestWavFile(self, filename, wav_data):
    with open(filename, 'wb') as f:
      f.write(wav_data)

  def _SaveWavFolders(self, root_dir, labels, how_many):
    wav_data = self._GetWavData()
    for label in labels:
      dir_name = os.path.join(root_dir, label)
      os.mkdir(dir_name)
      for i in range(how_many):
        file_path = os.path.join(dir_name, 'some_audio_%d.wav' % i)
        self._SaveTestWavFile(file_path, wav_data)

  def _PrepareDummyTrainingData(self):
    tmp_dir = FLAGS.test_tmpdir
    # create data folder with subfolders,
    # where every subfolder is a category/label with wav data inside
    # we will automatically split these data into
    # training, validation and testing sets
    data_dir = os.path.join(tmp_dir, 'data1')
    os.mkdir(data_dir)
    self._SaveWavFolders(data_dir, ['a', 'b', 'c'], 100)
    background_dir = os.path.join(data_dir, '_background_noise_')
    os.mkdir(background_dir)
    wav_data = self._GetWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, 'background_audio_%d.wav' % i)
      self._SaveTestWavFile(file_path, wav_data)
    return data_dir

  def _PrepareDummyTrainingDataSplit(self):
    tmp_dir = FLAGS.test_tmpdir

    # main wav data folder
    data_dir = os.path.join(tmp_dir, 'data0')
    os.mkdir(data_dir)

    # create 4 subfolders: training, validation, testing, _background_noise_
    # training data
    training_dir = os.path.join(data_dir, 'training')
    os.mkdir(training_dir)
    self._SaveWavFolders(training_dir, ['a', 'b', 'c'], 100)

    # validation data
    validation_dir = os.path.join(data_dir, 'validation')
    os.mkdir(validation_dir)
    self._SaveWavFolders(validation_dir, ['a', 'b', 'c'], 100)

    # testing data
    testing_dir = os.path.join(data_dir, 'testing')
    os.mkdir(testing_dir)
    self._SaveWavFolders(testing_dir, ['a', 'b', 'c'], 100)

    # _background_noise_ data
    background_dir = os.path.join(data_dir, '_background_noise_')
    os.mkdir(background_dir)
    wav_data = self._GetWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, 'background_audio_%d.wav' % i)
      self._SaveTestWavFile(file_path, wav_data)
    return data_dir

  def _PrepareDummyDir(self, dir_name):
    path = os.path.join(FLAGS.test_tmpdir, dir_name)
    os.mkdir(path)
    return path

  def _GetDefaultFlags(self, split_data):
    dummy_flags = {
        'data_url':
            '',
        'data_dir':
            self._PrepareDummyTrainingData()
            if split_data == 1 else self._PrepareDummyTrainingDataSplit(),
        'wanted_words':
            'a,b,c',
        'sample_rate':
            16000,
        'clip_duration_ms':
            1000,
        'window_size_ms':
            30,
        'window_stride_ms':
            20,
        'feature_bin_count':
            40,
        'preprocess':
            'raw',
        'split_data':
            split_data,
        'silence_percentage':
            25,
        'unknown_percentage':
            25,
        'validation_percentage':
            10,
        'testing_percentage':
            10,
        'summaries_dir':
            self._PrepareDummyDir('summaries' + str(split_data)),
        'train_dir':
            self._PrepareDummyDir('train' + str(split_data)),
        'time_shift_ms':
            100,
        'how_many_training_steps':
            '2',
        'learning_rate':
            '0.01',
        'quantize':
            False,
        'check_nans':
            False,
        'start_checkpoint':
            '',
        'model_params_path':
            '',
        'model_name':
            'dnn',
        'batch_size':
            1,
        'background_volume':
            0.25,
        'background_frequency':
            0.8,
        'eval_step_interval':
            1,
        'save_step_interval':
            1,
        'verbosity':
            logging.INFO,
        'optimizer_epsilon':
            1e-08,
        'log_epsilon':
            1e-12,
        'resample':
            0.0,
        'use_tf_fft':
            0,
        'preemph':
            0.0,
        'window_type':
            'hann',
        'mel_num_bins':
            40,
        'mel_lower_edge_hertz':
            20.0,
        'mel_upper_edge_hertz':
            4000.0,
        'fft_magnitude_squared':
            0,
        'mel_non_zero_only':
            1,
        'dct_num_features':
            10,
        'units1':
            '32',
        'act1':
            "'relu'",
        'pool_size':
            2,
        'strides':
            2,
        'dropout1':
            0.1,
        'units2':
            '256,256',
        'act2':
            "'relu','relu'",
    }
    return DictStruct(**dummy_flags)

  @parameterized.named_parameters([
      dict(testcase_name='default data split', split_data=1),
      dict(testcase_name='user splits data', split_data=0)
  ])
  def testTrain(self, split_data):
    input_flags = self._GetDefaultFlags(split_data)
    input_flags = model_flags.update_flags(input_flags)
    train.train(input_flags)
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(input_flags.train_dir, 'graph.pbtxt')))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(input_flags.train_dir, 'labels.txt')))
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(input_flags.train_dir, 'accuracy_last.txt')))


if __name__ == '__main__':
  tf.test.main()
