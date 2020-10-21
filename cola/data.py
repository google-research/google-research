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

"""Provides helper data related functions."""

import tensorflow as tf
import tensorflow_datasets as tfds

from cola import constants


def get_self_supervised_data(dataset=constants.Dataset.LBS,
                             shuffle_buffer=1000):
  """Reads TFDS data for self-supervised task."""

  def _parse_example(audio, _):
    return {"audio": tf.cast(audio, tf.float32) / float(tf.int16.max)}

  if dataset == constants.Dataset.LBS:
    split = "train_clean360"
  else:
    split = "train"

  ds_train = tfds.load(
      dataset.value, split=split, as_supervised=True)
  ds_train = ds_train.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
  ds_train = ds_train.map(
      _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return ds_train


def get_downstream_dataset(dataset=constants.Dataset.VOXFORGE,
                           shuffle_buffer=1000):
  """Reads downstream task data from TFDS."""

  def _parse_example(audio, label):
    audio = tf.cast(audio, tf.float32) / float(tf.int16.max)
    return {"audio": audio, "label": label}

  (ds_train, ds_test), ds_info = tfds.load(
      dataset.value,
      split=["train", "test"],
      shuffle_files=True,
      as_supervised=True,
      with_info=True)

  ds_train = ds_train.shuffle(
      shuffle_buffer, reshuffle_each_iteration=True).map(
          _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.shuffle(
      shuffle_buffer, reshuffle_each_iteration=True).map(
          _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return (ds_train, ds_test, ds_info.features["label"].num_classes)


def extract_log_mel_spectrogram(waveform,
                                sample_rate=16000,
                                frame_length=400,
                                frame_step=160,
                                fft_length=1024,
                                n_mels=64,
                                fmin=60.0,
                                fmax=7800.0):
  """Extract frames of log mel spectrogram from a raw waveform."""

  stfts = tf.signal.stft(
      waveform,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length)
  spectrograms = tf.abs(stfts)

  num_spectrogram_bins = stfts.shape[-1]
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  mel_spectrograms = tf.clip_by_value(
      mel_spectrograms,
      clip_value_min=1e-5,
      clip_value_max=1e8)

  log_mel_spectrograms = tf.math.log(mel_spectrograms)

  return log_mel_spectrograms


def extract_window(waveform, seg_length=16000):
  """Extracts a random segment from a waveform."""
  padding = tf.maximum(seg_length - tf.shape(waveform)[0], 0)
  left_pad = padding // 2
  right_pad = padding - left_pad
  padded_waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]])
  return tf.image.random_crop(padded_waveform, [seg_length])
