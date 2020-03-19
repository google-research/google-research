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

"""Simple speech recognition to spot a limited number of keywords.

It is based on tensorflow/examples/speech_commands
This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. This network uses a
keyword detection style to spot discrete words from a small vocabulary,
consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run model_train_eval.py

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, it will produce
Keras, SavedModel, TFLite and graphdef representations.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

data >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train --
--data_dir /data --wanted_words up,down

Above script will automatically split data into training/validation and testing.

If you prefer to split the data on your own, then you should set flag
"--split_data 0" and prepare folders with structure:

data >
  training >
    up >
      audio_0.wav
      audio_1.wav
    down >
      audio_2.wav
      audio_3.wav
  validation >
    up >
      audio_6.wav
      audio_7.wav
    down >
      audio_8.wav
      audio_9.wav
  testing >
    up >
      audio_12.wav
      audio_13.wav
    down >
      audio_14.wav
      audio_15.wav
  _background_noise_ >
    audio_18.wav

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train --
--data_dir /data --wanted_words up,down --split_data 0

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
from absl import logging
import tensorflow.compat.v1 as tf
from kws_streaming.layers.modes import Modes
import kws_streaming.models.att_mh_rnn as att_mh_rnn
import kws_streaming.models.att_rnn as att_rnn
import kws_streaming.models.cnn as cnn
import kws_streaming.models.crnn as crnn
import kws_streaming.models.dnn as dnn
import kws_streaming.models.dnn_raw as dnn_raw
import kws_streaming.models.ds_cnn as ds_cnn
import kws_streaming.models.gru as gru
import kws_streaming.models.lstm as lstm
import kws_streaming.models.svdf as svdf
from kws_streaming.train import model_flags
from kws_streaming.train import train
import kws_streaming.train.test as test


FLAGS = None


def main(_):
  # Update flags
  flags = model_flags.update_flags(FLAGS)

  if flags.train:
    # Create model folders where logs and model will be stored
    os.makedirs(flags.train_dir)
    os.mkdir(flags.summaries_dir)

    # Model training
    train.train(flags)

  # write all flags settings into json
  with open(os.path.join(flags.train_dir, 'flags.json'), 'wt') as f:
    json.dump(flags.__dict__, f)

  # convert to SavedModel
  test.convert_model_saved(flags, 'non_stream', Modes.NON_STREAM_INFERENCE)
  test.convert_model_saved(flags, 'stream_state_internal',
                           Modes.STREAM_INTERNAL_STATE_INFERENCE)

  # ---------------- non streaming model accuracy evaluation ----------------
  # with TF
  folder_name = 'tf'
  test.tf_non_stream_model_accuracy(flags, folder_name)

  # with TF.
  # We can apply non stream model on stream data, by running inference
  # every 200ms (for example), so that total latency will be similar with
  # streaming model which is executed every 20ms.
  # To measure the impact of sampling on model accuracy,
  # we introduce time_shift_ms during accuracy evaluation.
  # Convert milliseconds to samples:
  time_shift_samples = int(
      (flags.time_shift_ms * flags.sample_rate) / model_flags.MS_PER_SECOND)
  test.tf_non_stream_model_accuracy(
      flags,
      folder_name,
      time_shift_samples,
      accuracy_name='tf_non_stream_model_sampling_stream_accuracy.txt')

  name2opt = {
      '': None,
      'quantize_opt_for_size_': [tf.lite.Optimize.OPTIMIZE_FOR_SIZE],
  }

  for opt_name, optimizations in name2opt.items():
    folder_name = opt_name + 'tflite_non_stream'
    file_name = 'non_stream.tflite'
    mode = Modes.NON_STREAM_INFERENCE
    test.convert_model_tflite(flags, folder_name, mode, file_name,
                              optimizations=optimizations)
    test.tflite_non_stream_model_accuracy(flags, folder_name, file_name)

    # ---------------- TF streaming model accuracy evaluation ----------------
    # Streaming model (with external state) evaluation using TF with state reset
    if not opt_name:  # run TF evalution only without optimization/quantization
      folder_name = 'tf'
      test.tf_stream_state_external_model_accuracy(
          flags,
          folder_name,
          accuracy_name='stream_state_external_model_accuracy_sub_set_reset1.txt',
          reset_state=True)  # with state reset between test sequences

      # Streaming model (with external state) evaluation using TF no state reset
      test.tf_stream_state_external_model_accuracy(
          flags,
          folder_name,
          accuracy_name='stream_state_external_model_accuracy_sub_set_reset0.txt',
          reset_state=False)  # without state reset

      # Streaming model (with internal state) evaluation using TF no state reset
      test.tf_stream_state_internal_model_accuracy(flags, folder_name)

    # --------------- TFlite streaming model accuracy evaluation ---------------
    # convert model to TFlite
    folder_name = opt_name + 'tflite_stream_state_external'
    file_name = 'stream_state_external.tflite'
    mode = Modes.STREAM_EXTERNAL_STATE_INFERENCE
    test.convert_model_tflite(flags, folder_name, mode, file_name,
                              optimizations=optimizations)

    # Streaming model accuracy evaluation with TFLite with state reset
    test.tflite_stream_state_external_model_accuracy(
        flags,
        folder_name,
        file_name,
        accuracy_name='tflite_stream_state_external_model_accuracy_reset1.txt',
        reset_state=True)

    # Streaming model accuracy evaluation with TFLite without state reset
    test.tflite_stream_state_external_model_accuracy(
        flags,
        folder_name,
        file_name,
        accuracy_name='tflite_stream_state_external_model_accuracy_reset0.txt',
        reset_state=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--split_data',
      type=int,
      default=1,
      help="""\
      If 1, it will split data located in data_dir \
      into training/testing/validation data sets. \
      It will use flags: silence_percentage, unknown_percentage, \
      testing_percentage, validation_percentage. \
      In addition to categories in wanted_words it will add \
      two more categories _silence_ _unknown_ \
      If 0, it will read data from folders: \
      data_dir/training, data_dir/testing, data_dir/validation \
      Data in above folders have to be prepared by user. \
      It will ignore flags: silence_percentage, unknown_percentage, \
      testing_percentage, validation_percentage. \
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='600,600',
      help='How many training loops to run',
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',
  )
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')

  # Function used to parse --verbosity argument
  def verbosity_arg(value):
    """Parses verbosity argument.

    Args:
      value: A member of tf.logging.

    Returns:
      TF logging mode

    Raises:
      ArgumentTypeError: Not an expected value.
    """
    value = value.upper()
    if value == 'INFO':
      return logging.INFO
    elif value == 'DEBUG':
      return logging.DEBUG
    elif value == 'ERROR':
      return logging.ERROR
    elif value == 'FATAL':
      return logging.FATAL
    elif value == 'WARN':
      return logging.WARN
    else:
      raise argparse.ArgumentTypeError('Not an expected value')

  parser.add_argument(
      '--verbosity',
      type=verbosity_arg,
      default=logging.INFO,
      help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"')
  parser.add_argument(
      '--optimizer_epsilon',
      type=float,
      default=1e-08,
      help='Epsilon of Adam optimizer.',
  )
  parser.add_argument(
      '--resample',
      type=float,
      default=0.0,
      help='Resample input signal to generate more data [0.0...0.15].',
  )
  parser.add_argument(
      '--train',
      type=int,
      default=1,
      help='If 1 run train and test, else run only test',
  )

  # speech feature extractor properties
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',
  )
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',
  )
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=40.0,
      help='How long each spectrogram timeslice is.',
  )
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=20.0,
      help='How far to move in time between spectrogram timeslices.',
  )
  parser.add_argument(
      '--preprocess',
      type=str,
      default='raw',
      help='Supports only raw')
  parser.add_argument(
      '--feature_type',
      type=str,
      default='mfcc_tf',
      help='Feature type: mfcc_tf, mfcc_op')
  parser.add_argument(
      '--preemph',
      type=float,
      default=0.0,
      help='Preemphasis filter standard value 0.97, '
      'if 0.0 - preemphasis will disabled',
  )
  parser.add_argument(
      '--window_type',
      type=str,
      default='hann',
      help='Window type used on input frame before computing Fourier Transform',
  )
  parser.add_argument(
      '--mel_lower_edge_hertz',
      type=float,
      default=20.0,
      help='Lower bound on the frequencies to be included in the mel spectrum.'
      'This corresponds to the lower edge of the lowest triangular band.',
  )
  parser.add_argument(
      '--mel_upper_edge_hertz',
      type=float,
      default=4000.0,
      help='The desired top edge of the highest frequency band',
  )
  parser.add_argument(
      '--log_epsilon',
      type=float,
      default=1e-12,
      help='epsilon for log function in speech feature extractor',
  )
  parser.add_argument(
      '--dct_num_features',
      type=int,
      default=10,
      help='Number of features left after DCT',
  )
  parser.add_argument(
      '--use_tf_fft',
      type=int,
      default=0,
      help='if 1 we will use TF FFT otherwise use direct DFT which is'
      'implemented using matrix matrix multiplications and supported by'
      'any inference engine',
  )
  parser.add_argument(
      '--mel_non_zero_only',
      type=int,
      default=1,
      help='if 1 we will check non zero range of mel spectrum and use it'
      'to reduce DFT computation, otherwise full DFT is computed, '
      'it can not be used together with use_tf_fft')
  parser.add_argument(
      '--fft_magnitude_squared',
      type=int,
      default=0,
      help='if 1 magnitude spectrum will be squared otherwise sqrt',
  )
  parser.add_argument(
      '--mel_num_bins',
      type=int,
      default=40,
      help='How many bands in the resulting mel spectrum.',
  )

  subparsers = parser.add_subparsers(dest='model_name', help='NN model name')

  # DNN model settings
  parser_dnn = subparsers.add_parser('dnn')
  dnn.model_parameters(parser_dnn)

  # DNN raw model settings
  parser_dnn_raw = subparsers.add_parser('dnn_raw')
  dnn_raw.model_parameters(parser_dnn_raw)

  # LSTM model settings
  parser_lstm = subparsers.add_parser('lstm')
  lstm.model_parameters(parser_lstm)

  # GRU model settings
  parser_gru = subparsers.add_parser('gru')
  gru.model_parameters(parser_gru)

  # SVDF model settings
  parser_svdf = subparsers.add_parser('svdf')
  svdf.model_parameters(parser_svdf)

  # CNN model settings
  parser_cnn = subparsers.add_parser('cnn')
  cnn.model_parameters(parser_cnn)

  # CRNN model settings
  parser_crnn = subparsers.add_parser('crnn')
  crnn.model_parameters(parser_crnn)

  # ATT MH RNN model settings
  parser_att_mh_rnn = subparsers.add_parser('att_mh_rnn')
  att_mh_rnn.model_parameters(parser_att_mh_rnn)

  # ATT RNN model settings
  parser_att_rnn = subparsers.add_parser('att_rnn')
  att_rnn.model_parameters(parser_att_rnn)

  # DS_CNN model settings
  parser_ds_cnn = subparsers.add_parser('ds_cnn')
  ds_cnn.model_parameters(parser_ds_cnn)

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
