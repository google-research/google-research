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

"""Data reader, based on tensorflow/examples/speech_commands."""

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
from absl import logging
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
from kws_streaming.layers import modes

# pylint: disable=g-direct-tensorflow-import
# below ops are on a depreciation path in tf, so we temporarily disable pylint
# to be able to import them: TODO(rybakov) - use direct tf

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

tf.disable_eager_execution()

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
  from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
  frontend_op = None
# pylint: enable=g-direct-tensorflow-import

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185
MAX_ABS_INT16 = 32768


def prepare_words_list(wanted_words, split_data):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.
    split_data: True - split data automatically; False - user splits the data

  Returns:
    List with the standard silence and unknown tokens added.
  """
  if split_data:
    # with automatic data split we append two more labels
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words
  else:
    # data already split by user, no need to add other labels
    return wanted_words


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


def load_wav_file(filename):
  """Loads an audio file and returns a float PCM-encoded array of samples.

  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
  """Saves audio sample data to a .wav audio file.

  Args:
    filename: Path to save the file to.
    wav_data: 2D array of float PCM-encoded audio data.
    sample_rate: Samples per second to encode in the file.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    sample_rate_placeholder = tf.placeholder(tf.int32, [])
    wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
    wav_encoder = tf.audio.encode_wav(wav_data_placeholder,
                                      sample_rate_placeholder)
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data.

    Args:
      flags: data and model parameters, described at model_train_eval.py
  """

  def __init__(self, flags):
    wanted_words = flags.wanted_words.split(',')
    if flags.data_dir:
      self.data_dir = flags.data_dir
      if flags.split_data:
        self.maybe_download_and_extract_dataset(flags.data_url, self.data_dir)
        self.prepare_data_index(flags.silence_percentage,
                                flags.unknown_percentage, wanted_words,
                                flags.validation_percentage,
                                flags.testing_percentage, flags.split_data)
      else:
        self.prepare_split_data_index(wanted_words, flags.split_data)

      self.prepare_background_data()
    self.prepare_processing_graph(flags)

  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        logging.error(
            'Failed to download URL: %s to folder: %s\n'
            'Please make sure you have enough free space and'
            ' an internet connection', data_url, filepath)
        raise
      print()
      statinfo = os.stat(filepath)
      logging.info('Successfully downloaded %s (%d bytes)', filename,
                   statinfo.st_size)
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage, split_data):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.
      split_data: True - split data automatically; False - user splits the data

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      if word == BACKGROUND_NOISE_DIR_NAME:
        continue
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words, split_data)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  def validate_dir_structure(self, data_dir, dirs):
    for dir_name in dirs + [BACKGROUND_NOISE_DIR_NAME]:
      sub_dir_name = os.path.join(data_dir, dir_name)
      if not os.path.isdir(sub_dir_name):
        raise IOError('Directory is not found ' + sub_dir_name)

  def prepare_split_data_index(self, wanted_words, split_data):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`,
    where `data_dir` has to contain folders (prepared by user):
      testing
      training
      validation
      _background_noise_ - contains data which are used for adding background
      noise to training data only

    Args:
      wanted_words: Labels of the classes we want to be able to recognize.
      split_data: True - split data automatically; False - user splits the data

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)

    dirs = ['testing', 'training', 'validation']

    self.validate_dir_structure(self.data_dir, dirs)

    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index

    self.words_list = prepare_words_list(wanted_words, split_data)

    self.data_index = {'validation': [], 'testing': [], 'training': []}

    for set_index in dirs:
      all_words = {}
      # Look through all the subfolders in set_index to find audio samples
      search_path = os.path.join(
          os.path.join(self.data_dir, set_index), '*', '*.wav')
      for wav_path in gfile.Glob(search_path):
        _, word = os.path.split(os.path.dirname(wav_path))
        word = word.lower()
        # Treat the '_background_noise_' folder as a special case,
        # it contains long audio samples we mix in to improve training.
        if word == BACKGROUND_NOISE_DIR_NAME:
          continue
        all_words[word] = True

        # If it's a known class, store its detail, otherwise raise error
        if word in wanted_words_index:
          self.data_index[set_index].append({'label': word, 'file': wav_path})
        else:
          raise Exception('Unknown word ' + word)

      if not all_words:
        raise IOError('No .wavs found at ' + search_path)
      for index, wanted_word in enumerate(wanted_words):
        if wanted_word not in all_words:
          raise IOError('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))

    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])

    # Prepare the rest of the result data structure.
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        raise Exception('Unknown word ' + word)

  def prepare_background_data(self):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
      return self.background_data
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
      search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                 '*.wav')
      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        self.background_data.append(wav_data)
      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)

  def prepare_processing_graph(self, flags):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - foreground_resampling_placeholder_: Controls signal stretching/squeezing
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - output_: Output 2D fingerprint of processed audio or raw audio.

    Args:
      flags: data and model parameters, described at model_train.py

    Raises:
      ValueError: If the preprocessing mode isn't recognized.
      Exception: If the preprocessor wasn't compiled in.
    """
    with tf.get_default_graph().name_scope('data'):
      desired_samples = flags.desired_samples
      self.wav_filename_placeholder_ = tf.placeholder(
          tf.string, [], name='wav_filename')
      wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
      wav_decoder = tf.audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)

      # Allow the audio sample's volume to be adjusted.
      self.foreground_volume_placeholder_ = tf.placeholder(
          tf.float32, [], name='foreground_volume')
      # signal resampling to generate more training data
      # it will stretch or squeeze input signal proportinally to:
      self.foreground_resampling_placeholder_ = tf.placeholder(tf.float32, [])

      if self.foreground_resampling_placeholder_ != 1.0:
        image = tf.expand_dims(wav_decoder.audio, 0)
        image = tf.expand_dims(image, 2)
        shape = tf.shape(wav_decoder.audio)
        image_resized = tf.image.resize(
            images=image,
            size=(tf.cast((tf.cast(shape[0], tf.float32) *
                           self.foreground_resampling_placeholder_),
                          tf.int32), 1),
            preserve_aspect_ratio=False)
        image_resized_cropped = tf.image.resize_with_crop_or_pad(
            image_resized,
            target_height=desired_samples,
            target_width=1,
        )
        image_resized_cropped = tf.squeeze(image_resized_cropped, axis=[0, 3])
        scaled_foreground = tf.multiply(image_resized_cropped,
                                        self.foreground_volume_placeholder_)
      else:
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                        self.foreground_volume_placeholder_)
      # Shift the sample's start position, and pad any gaps with zeros.
      self.time_shift_padding_placeholder_ = tf.placeholder(
          tf.int32, [2, 2], name='time_shift_padding')
      self.time_shift_offset_placeholder_ = tf.placeholder(
          tf.int32, [2], name='time_shift_offset')
      padded_foreground = tf.pad(
          tensor=scaled_foreground,
          paddings=self.time_shift_padding_placeholder_,
          mode='CONSTANT')
      sliced_foreground = tf.slice(padded_foreground,
                                   self.time_shift_offset_placeholder_,
                                   [desired_samples, -1])
      # Mix in background noise.
      self.background_data_placeholder_ = tf.placeholder(
          tf.float32, [desired_samples, 1], name='background_data')
      self.background_volume_placeholder_ = tf.placeholder(
          tf.float32, [], name='background_volume')
      background_mul = tf.multiply(self.background_data_placeholder_,
                                   self.background_volume_placeholder_)
      background_add = tf.add(background_mul, sliced_foreground)
      background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

      if flags.preprocess == 'raw':
        # background_clamp dims: [time, channels]
        # remove channel dim
        self.output_ = tf.squeeze(background_clamp, axis=1)
      # below options are for backward compatibility with previous
      # version of hotword detection on microcontrollers
      # in this case audio feature extraction is done separately from
      # neural net and user will have to manage it.
      elif flags.preprocess == 'mfcc':
        # Run the spectrogram and MFCC ops to get a 2D audio: Short-time FFTs
        # background_clamp dims: [time, channels]
        spectrogram = audio_ops.audio_spectrogram(
            background_clamp,
            window_size=flags.window_size_samples,
            stride=flags.window_stride_samples,
            magnitude_squared=flags.fft_magnitude_squared)
        # spectrogram: [channels/batch, frames, fft_feature]

        # extract mfcc features from spectrogram by audio_ops.mfcc:
        # 1 Input is spectrogram frames.
        # 2 Weighted spectrogram into bands using a triangular mel filterbank
        # 3 Logarithmic scaling
        # 4 Discrete cosine transform (DCT), return lowest dct_coefficient_count
        mfcc = audio_ops.mfcc(
            spectrogram=spectrogram,
            sample_rate=flags.sample_rate,
            upper_frequency_limit=flags.mel_upper_edge_hertz,
            lower_frequency_limit=flags.mel_lower_edge_hertz,
            filterbank_channel_count=flags.mel_num_bins,
            dct_coefficient_count=flags.dct_num_features)
        # mfcc: [channels/batch, frames, dct_coefficient_count]
        # remove channel dim
        self.output_ = tf.squeeze(mfcc, axis=0)
      elif flags.preprocess == 'micro':
        if not frontend_op:
          raise Exception(
              'Micro frontend op is currently not available when running'
              ' TensorFlow directly from Python, you need to build and run'
              ' through Bazel')
        int16_input = tf.cast(
            tf.multiply(background_clamp, MAX_ABS_INT16), tf.int16)
        # audio_microfrontend does:
        # 1. A slicing window function of raw audio
        # 2. Short-time FFTs
        # 3. Filterbank calculations
        # 4. Noise reduction
        # 5. PCAN Auto Gain Control
        # 6. Logarithmic scaling

        # int16_input dims: [time, channels]
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=flags.sample_rate,
            window_size=flags.window_size_ms,
            window_step=flags.window_stride_ms,
            num_channels=flags.mel_num_bins,
            upper_band_limit=flags.mel_upper_edge_hertz,
            lower_band_limit=flags.mel_lower_edge_hertz,
            out_scale=1,
            out_type=tf.float32)
        # int16_input dims: [frames, num_channels]
        self.output_ = tf.multiply(micro_frontend, (10.0 / 256.0))
      else:
        raise ValueError('Unknown preprocess mode "%s" (should be "raw", '
                         ' "mfcc", or "micro")' % (flags.preprocess))

  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, flags, background_frequency,
               background_volume_range, time_shift, mode, resample_offset,
               volume_augmentation_offset, sess):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      flags: data and model parameters, described at model_train.py
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift: How much to randomly shift the clips by in time.
        It shifts audio data in range from -time_shift to time_shift.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
      resample_offset: resample input signal - stretch it or squeeze by 0..0.15
        If 0 - then not resampling.
      volume_augmentation_offset: it is used for raw audio volume control.
        During training volume multiplier will be sampled from
        1.0 - volume_augmentation_offset ... 1.0 + volume_augmentation_offset
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of label indexes

    Raises:
      ValueError: If background samples are too short.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      if flags.pick_deterministically and mode == 'training':
        # it is a special case:
        sample_count = how_many
      else:
        sample_count = max(0, min(how_many, len(candidates) - offset))

    # Data and labels will be populated and returned.
    input_data_shape = modes.get_input_data_shape(flags, modes.Modes.TRAINING)
    data = np.zeros((sample_count,) + input_data_shape)
    labels = np.zeros(sample_count)
    desired_samples = flags.desired_samples
    use_background = self.background_data and (mode == 'training')
    pick_deterministically = (mode !=
                              'training') or flags.pick_deterministically
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        # during inference offset is 0,
        # but during training offset can be 0 or
        # training_step * batch_size, so 'i' can go beyond array size
        sample_index = i % len(candidates)
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
      # If we're time shifting, set up the offset for this sample.
      if time_shift > 0:
        time_shift_amount = np.random.randint(-time_shift, time_shift)
      else:
        time_shift_amount = 0
      if time_shift_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
      else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]

      resample = 1.0
      if mode == 'training' and resample_offset != 0.0:
        resample = np.random.uniform(
            low=resample - resample_offset, high=resample + resample_offset)
      input_dict = {
          self.wav_filename_placeholder_: sample['file'],
          self.time_shift_padding_placeholder_: time_shift_padding,
          self.time_shift_offset_placeholder_: time_shift_offset,
          self.foreground_resampling_placeholder_: resample,
      }
      # Choose a section of background noise to mix in.
      if use_background:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        if len(background_samples) <= flags.desired_samples:
          raise ValueError('Background sample is too short! Need more than %d'
                           ' samples but only %d were found' %
                           (flags.desired_samples, len(background_samples)))
        background_offset = np.random.randint(
            0,
            len(background_samples) - flags.desired_samples)
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        background_reshaped = background_clipped.reshape([desired_samples, 1])
        if np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random.uniform(0, background_volume_range)
        else:
          background_volume = 0
      else:
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0
      input_dict[self.background_data_placeholder_] = background_reshaped
      input_dict[self.background_volume_placeholder_] = background_volume
      # If we want silence, mute out the main sample but leave the background.
      if sample['label'] == SILENCE_LABEL:
        input_dict[self.foreground_volume_placeholder_] = 0
      else:
        foreground_volume = 1.0  # multiplier of audio signal
        # in training mode produce audio data with different volume
        if mode == 'training' and volume_augmentation_offset != 0.0:
          foreground_volume = np.random.uniform(
              low=foreground_volume - volume_augmentation_offset,
              high=foreground_volume + volume_augmentation_offset)

        input_dict[self.foreground_volume_placeholder_] = foreground_volume
      # Run the graph to produce the output audio.
      data_tensor = sess.run(self.output_, feed_dict=input_dict)
      data[i - offset, :] = data_tensor
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
    return data, labels

  def get_features_for_wav(self, wav_filename, flags, sess):
    """Applies the feature transformation process to the input_wav.

    Runs the feature generation process (generally producing a spectrogram from
    the input samples) on the WAV file. This can be useful for testing and
    verifying implementations being run on other platforms.

    Args:
      wav_filename: The path to the input audio file.
      flags: data and model parameters, described at model_train.py
      sess: TensorFlow session that was active when processor was created.

    Returns:
      Numpy data array containing the generated features.
    """
    desired_samples = flags.desired_samples
    input_dict = {
        self.wav_filename_placeholder_: wav_filename,
        self.time_shift_padding_placeholder_: [[0, 0], [0, 0]],
        self.time_shift_offset_placeholder_: [0, 0],
        self.background_data_placeholder_: np.zeros([desired_samples, 1]),
        self.background_volume_placeholder_: 0,
        self.foreground_volume_placeholder_: 1,
        self.foreground_resampling_placeholder_: 1.0,
    }
    # Run the graph to produce the output audio.
    data_tensor = sess.run([self.output_], feed_dict=input_dict)
    return data_tensor

  def get_unprocessed_data(self, how_many, flags, mode):
    """Retrieve sample data for the given partition, with no transformations.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      flags: data and model parameters, described at model_train.py
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.

    Returns:
      List of sample data for the samples, and list of labels in one-hot form.
    """
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many
    desired_samples = flags.desired_samples
    words_list = self.words_list
    data = np.zeros((sample_count, desired_samples))
    labels = []
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = tf.audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      foreground_volume_placeholder = tf.placeholder(tf.float32, [])
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      foreground_volume_placeholder)
      for i in range(sample_count):
        if how_many == -1:
          sample_index = i
        else:
          sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]
        input_dict = {wav_filename_placeholder: sample['file']}
        if sample['label'] == SILENCE_LABEL:
          input_dict[foreground_volume_placeholder] = 0
        else:
          input_dict[foreground_volume_placeholder] = 1
        data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
        label_index = self.word_to_index[sample['label']]
        labels.append(words_list[label_index])
    return data, labels
