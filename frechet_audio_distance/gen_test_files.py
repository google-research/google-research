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

"""Creates a set of audio files to test FAD calculation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os

from absl import app
from absl import flags

import numpy as np
import scipy.io.wavfile

_SAMPLE_RATE = 16000

FLAGS = flags.FLAGS
flags.DEFINE_string("test_files", "",
                    "Directory where the test files should be located")


def create_dir(output_dir):
  """Ignore directory exists error."""
  try:
    os.makedirs(output_dir)
  except OSError as exception:
    if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
      pass
    else:
      raise


def add_noise(data, stddev):
  """Adds Gaussian noise to the samples.

  Args:
    data: 1d Numpy array containing floating point samples. Not necessarily
      normalized.
    stddev: The standard deviation of the added noise.

  Returns:
     1d Numpy array containing the provided floating point samples with added
     Gaussian noise.

  Raises:
    ValueError: When data is not a 1d numpy array.
  """
  if len(data.shape) != 1:
    raise ValueError("expected 1d numpy array.")
  max_value = np.amax(np.abs(data))
  num_samples = data.shape[0]
  gauss = np.random.normal(0, stddev, (num_samples)) * max_value
  return data + gauss


def gen_sine_wave(freq=600,
                  length_seconds=6,
                  sample_rate=_SAMPLE_RATE,
                  param=None):
  """Creates sine wave of the specified frequency, sample_rate and length."""
  t = np.linspace(0, length_seconds, int(length_seconds * sample_rate))
  samples = np.sin(2 * np.pi * t * freq)
  if param:
    samples = add_noise(samples, param)
  return np.asarray(2**15 * samples, dtype=np.int16)


def main(argv):
  del argv  # Unused.
  for traget, count, param in [("background", 10, None), ("test1", 5, 0.0001),
                               ("test2", 5, 0.00001)]:
    output_dir = os.path.join(FLAGS.test_files, traget)
    create_dir(output_dir)
    print("output_dir:", output_dir)
    frequencies = np.linspace(100, 1000, count).tolist()
    for freq in frequencies:
      samples = gen_sine_wave(freq, param=param)
      filename = os.path.join(output_dir, "sin_%.0f.wav" % freq)
      print("Creating: %s with %i samples." % (filename, samples.shape[0]))
      scipy.io.wavfile.write(filename, _SAMPLE_RATE, samples)


if __name__ == "__main__":
  app.run(main)
