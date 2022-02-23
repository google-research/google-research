# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Frontend for features."""

from typing import Any, Dict, Optional

from absl import logging
import tensorflow as tf
from non_semantic_speech_benchmark.export_model import tf_frontend


class SamplesToFeats(tf.keras.layers.Layer):
  """Compute features from samples."""

  def __init__(self, frontend_args = None):
    super(SamplesToFeats, self).__init__()
    self.frontend_args = frontend_args or {}

    def _default(k, v):
      if k not in self.frontend_args:
        self.frontend_args[k] = v

    _default('num_mel_bins', 80)
    _default('n_required', 32000)
    _default('frame_width', 195)
    _default('frame_hop', 195)
    _default('pad_mode', 'SYMMETRIC')
    logging.info('[SamplesToFeats] frontend_args: %s', self.frontend_args)

  def call(self, samples):
    if samples.shape.ndims != 2:
      raise ValueError(f'Frontend input must be ndim 2: {samples.shape.ndims}')
    # TODO(joelshor): When we get a frontend function that works on batches,
    # remove this loop.
    map_fn = lambda s: tf_frontend.compute_frontend_features(  # pylint:disable=g-long-lambda
        s,
        sr=16000,
        tflite=False,
        **self.frontend_args)
    return tf.map_fn(map_fn, samples, dtype=tf.float64)
