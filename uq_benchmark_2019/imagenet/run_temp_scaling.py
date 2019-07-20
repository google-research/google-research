# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python2, python3
"""Binary for running temperature scaling, writing temperature param to disk."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf

from uq_benchmark_2019 import array_utils
from uq_benchmark_2019 import calibration_lib
from uq_benchmark_2019 import metrics_lib
from uq_benchmark_2019 import uq_utils

gfile = tf.io.gfile
FLAGS = flags.FLAGS

NUM_EXAMPLES = 20000


def _declare_flags():
  """Declare flags; not invoked when this module is imported as a library."""
  flags.DEFINE_string('prediction_path', None, 'Path to predictions file.')


def run(prediction_path):
  """Run temperature scaling."""
  stats = array_utils.load_stats_from_tfrecords(prediction_path)
  probs = stats['probs'].astype(np.float32)
  labels = stats['labels'].astype(np.int32)
  if len(labels.shape) > 1:
    labels = np.squeeze(labels, -1)

  if probs.shape[0] > NUM_EXAMPLES:
    probs = probs[:NUM_EXAMPLES, :]
    labels = labels[:NUM_EXAMPLES]

  probs = metrics_lib.soften_probabilities(probs=probs)
  logits = uq_utils.np_inverse_softmax(probs)
  temp = calibration_lib.find_scaling_temperature(labels, logits)
  with gfile.GFile(
      os.path.join(os.path.dirname(prediction_path),
                   'temperature_hparam.json'), 'w') as fh:
    fh.write(json.dumps({'temperature': temp}))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.enable_v2_behavior()
  run(FLAGS.prediction_path)


if __name__ == '__main__':
  _declare_flags()
  app.run(main)
