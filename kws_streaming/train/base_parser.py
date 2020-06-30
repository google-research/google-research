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

"""Base parser with training/testing data speech features flags ."""


import argparse
from absl import logging


def base_parser():
  """Base parser.

  Flags parsing is splitted into two parts:
  1) base parser for training/testing data and speech flags, defined here.
  2) model parameters parser - it is defined in model file.

  Returns:
    parser
  """
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
      '--lr_schedule',
      type=str,
      default='linear',
      help="""\
      Learning rate schedule: linear, exp.
      """)
  parser.add_argument(
      '--optimizer',
      type=str,
      default='adam',
      help='Optimizer: adam, momentum')
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--l2_weight_decay',
      type=float,
      default=0.0,
      help="""\
      l2 weight decay for layers weights regularization.
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
      default='10000,10000,10000',
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
      default='0.0005,0.0001,0.00002',
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
      default=0.15,
      help='Resample input signal to generate more data [0.0...0.15].',
  )
  parser.add_argument(
      '--volume_resample',
      type=float,
      default=0.0,
      help='Controlls audio volume, '
      'sampled in range: [1-volume_resample...1+volume_resample].',
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
      help='Expected duration in milliseconds of the input wavs',
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
      help='Supports raw, mfcc, micro as input features for neural net'
      'raw - model is built end to end '
      'mfcc - model divided into mfcc feature extractor and neural net.'
      'micro - model divided into micro feature extractor and neural net.'
      'if mfcc/micro is selected user has to manage speech feature extractor '
      'and feed extracted features into neural net on device.'
      )
  parser.add_argument(
      '--feature_type',
      type=str,
      default='mfcc_tf',
      help='It is used only for model which uses preprocess=raw'
      'In this case model will contain speech feature extractor and neural net.'
      'It simplifies model deployment on device.'
      'Two options supported:'
      '* mfcc_tf, it will produce unquantized model only'
      '* mfcc_op, it will produce both quantized and unquantized models')
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
      default=7000.0,
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
      default=20,
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
  parser.add_argument(
      '--use_spec_augment',
      type=int,
      default=0,
      help='use SpecAugment',
  )
  parser.add_argument(
      '--time_masks_number',
      type=int,
      default=2,
      help='SpecAugment parameter time_masks_number',
  )
  parser.add_argument(
      '--time_mask_max_size',
      type=int,
      default=10,
      help='SpecAugment parameter time_mask_max_size.',
  )
  parser.add_argument(
      '--frequency_masks_number',
      type=int,
      default=2,
      help='SpecAugment parameter frequency_masks_number.',
  )
  parser.add_argument(
      '--frequency_mask_max_size',
      type=int,
      default=5,
      help='SpecAugment parameter frequency_mask_max_size.',
  )
  parser.add_argument(
      '--return_softmax',
      type=int,
      default=0,
      help='Use softmax in the model: '
      ' 0 for SparseCategoricalCrossentropy '
      ' 1 for CategoricalCrossentropy '
  )
  return parser
