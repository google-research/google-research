# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Test data reader, based on tensorflow/examples/speech_commands."""

import os
import numpy as np
import tensorflow.compat.v1 as tf
from kws_streaming.data import input_data
import kws_streaming.data.input_data_utils as du
from kws_streaming.models import model_flags
tf.disable_eager_execution()


# Used to convert a dictionary into an object, for mocking parsed flags.
class DictStruct(object):

  def __init__(self, **entries):
    self.__dict__.update(entries)


class InputDataTest(tf.test.TestCase):

  def _GetWavData(self):
    sample_data = tf.zeros([32000, 2])
    wav_encoder = tf.audio.encode_wav(sample_data, 16000)
    wav_data = self.evaluate(wav_encoder)
    return wav_data

  def _SaveTestWavFile(self, filename, wav_data):
    with open(filename, "wb") as f:
      f.write(wav_data)

  def _SaveWavFolders(self, root_dir, labels, how_many):
    wav_data = self._GetWavData()
    for label in labels:
      dir_name = os.path.join(root_dir, label)
      os.mkdir(dir_name)
      for i in range(how_many):
        file_path = os.path.join(dir_name, "some_audio_%d.wav" % i)
        self._SaveTestWavFile(file_path, wav_data)

  def _GetDefaultFlags(self):
    dummy_flags = {
        "desired_samples": 160,
        "label_count": 4,
        "sample_rate": 16000,
        "clip_duration_ms": 1000,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "window_size_ms": 40,
        "window_stride_ms": 20,
        "preprocess": "mfcc",
        "data_url": "",
        "validation_percentage": 10,
        "testing_percentage": 10,
        "wanted_words": "a,b",
        "silence_percentage": 10,
        "unknown_percentage": 10,
        "fft_magnitude_squared": False,
        "mel_upper_edge_hertz": 7000.0,
        "mel_lower_edge_hertz": 20.0,
        "mel_num_bins": 40,
        "dct_num_features": 30,
        "split_data": 1,
        "train": 1,
        "pick_deterministically": 0,
        "causal_data_frame_padding": 0,
        "wav": 1,
        "micro_enable_pcan": True,
        "micro_features_scale": (10. / 256.0),
        "micro_min_signal_remaining": 0.05,
        "micro_out_scale": 1,
    }
    return DictStruct(**dummy_flags)

  def _RunGetDataTest(self, preprocess, window_size_ms):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._SaveWavFolders(wav_dir, ["a", "b", "c"], 100)
    background_dir = os.path.join(wav_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._GetWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._SaveTestWavFile(file_path, wav_data)
    flags = self._GetDefaultFlags()
    flags.window_size_ms = window_size_ms
    flags.preprocess = preprocess
    flags.train_dir = tmp_dir
    flags.data_dir = wav_dir
    flags = model_flags.update_flags(flags)
    with self.cached_session() as sess:
      audio_processor = input_data.AudioProcessor(flags)
      result_data, result_labels = audio_processor.get_data(
          10, 0, flags, 0.3, 0.1, 100, "training", 0.0, 0.0, sess)

      self.assertLen(result_data, 10)
      self.assertLen(result_labels, 10)

  def testPrepareWordsList(self):
    words_list = ["a", "b"]
    self.assertGreater(
        len(du.prepare_words_list(words_list, split_data=True)),
        len(words_list))

  def testWhichSet(self):
    self.assertEqual(
        du.which_set("foo.wav", 10, 10),
        du.which_set("foo.wav", 10, 10))
    self.assertEqual(
        du.which_set("foo_nohash_0.wav", 10, 10),
        du.which_set("foo_nohash_1.wav", 10, 10))

  def testPrepareDataIndex(self):
    tmp_dir = self.get_temp_dir()
    self._SaveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    flags = self._GetDefaultFlags()
    flags.train_dir = tmp_dir
    flags.summaries_dir = tmp_dir
    flags.data_dir = tmp_dir
    audio_processor = input_data.AudioProcessor(flags)
    self.assertLess(0, audio_processor.set_size("training"))
    self.assertIn("training", audio_processor.data_index)
    self.assertIn("validation", audio_processor.data_index)
    self.assertIn("testing", audio_processor.data_index)
    self.assertEqual(du.UNKNOWN_WORD_INDEX,
                     audio_processor.word_to_index["c"])

  def testPrepareDataIndexEmpty(self):
    tmp_dir = self.get_temp_dir()
    self._SaveWavFolders(tmp_dir, ["a", "b", "c"], 0)
    with self.assertRaises(Exception) as e:
      flags = self._GetDefaultFlags()
      flags.train_dir = tmp_dir
      flags.summaries_dir = tmp_dir
      flags.data_dir = tmp_dir
      _ = input_data.AudioProcessor(flags)
    self.assertIn("No .wavs found", str(e.exception))

  def testPrepareDataIndexMissing(self):
    tmp_dir = self.get_temp_dir()
    self._SaveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    with self.assertRaises(Exception) as e:
      flags = self._GetDefaultFlags()
      flags.train_dir = tmp_dir
      flags.summaries_dir = tmp_dir
      flags.data_dir = tmp_dir
      flags.wanted_words = "a,b,d"
      _ = input_data.AudioProcessor(flags)
    self.assertIn("Expected to find", str(e.exception))

  def testPrepareBackgroundData(self):
    tmp_dir = self.get_temp_dir()
    background_dir = os.path.join(tmp_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._GetWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._SaveTestWavFile(file_path, wav_data)
    self._SaveWavFolders(tmp_dir, ["a", "b", "c"], 100)
    flags = self._GetDefaultFlags()
    flags.train_dir = tmp_dir
    flags.summaries_dir = tmp_dir
    flags.data_dir = tmp_dir
    audio_processor = input_data.AudioProcessor(flags)
    self.assertLen(audio_processor.background_data, 10)

  def testLoadWavFile(self):
    tmp_dir = self.get_temp_dir()
    file_path = os.path.join(tmp_dir, "load_test.wav")
    wav_data = self._GetWavData()
    self._SaveTestWavFile(file_path, wav_data)
    sample_data = input_data.load_wav_file(file_path)
    self.assertIsNotNone(sample_data)

  def testSaveWavFile(self):
    tmp_dir = self.get_temp_dir()
    file_path = os.path.join(tmp_dir, "load_test.wav")
    save_data = np.zeros([16000, 1])
    input_data.save_wav_file(file_path, save_data, 16000)
    loaded_data = input_data.load_wav_file(file_path)
    self.assertIsNotNone(loaded_data)
    self.assertLen(loaded_data, 16000)

  def testPrepareProcessingGraph(self):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._SaveWavFolders(wav_dir, ["a", "b", "c"], 100)
    background_dir = os.path.join(wav_dir, "_background_noise_")
    os.mkdir(background_dir)
    wav_data = self._GetWavData()
    for i in range(10):
      file_path = os.path.join(background_dir, "background_audio_%d.wav" % i)
      self._SaveTestWavFile(file_path, wav_data)
    flags = self._GetDefaultFlags()
    flags.train_dir = tmp_dir
    flags.summaries_dir = tmp_dir
    flags.data_dir = wav_dir
    audio_processor = input_data.AudioProcessor(flags)

    self.assertIsNotNone(audio_processor.wav_filename_placeholder_)
    self.assertIsNotNone(audio_processor.foreground_volume_placeholder_)
    self.assertIsNotNone(audio_processor.time_shift_padding_placeholder_)
    self.assertIsNotNone(audio_processor.time_shift_offset_placeholder_)
    self.assertIsNotNone(audio_processor.background_data_placeholder_)
    self.assertIsNotNone(audio_processor.background_volume_placeholder_)
    self.assertIsNotNone(audio_processor.output_)

  def testGetDataMfcc(self):
    self._RunGetDataTest("mfcc", 30)

  def testGetDataMicro(self):
    self._RunGetDataTest("micro", 20)

  def testGetUnprocessedData(self):
    tmp_dir = self.get_temp_dir()
    wav_dir = os.path.join(tmp_dir, "wavs")
    os.mkdir(wav_dir)
    self._SaveWavFolders(wav_dir, ["a", "b", "c"], 100)
    flags = self._GetDefaultFlags()
    flags.train_dir = tmp_dir
    flags.summaries_dir = tmp_dir
    flags.data_dir = wav_dir
    audio_processor = input_data.AudioProcessor(flags)
    result_data, result_labels = audio_processor.get_unprocessed_data(
        10, flags, "training")
    self.assertLen(result_data, 10)
    self.assertLen(result_labels, 10)


if __name__ == "__main__":
  tf.test.main()
