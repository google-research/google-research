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

"""Tests for tf.data based input pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import tensorflow.compat.v1 as tf

from qanet.util import configurable
from qanet import embedding
from qanet import experiment
from qanet.data import tf_data_pipeline

data_path = 'qanet/testdata'


class ResampleTest(tf.test.TestCase):

  def test_resample(self):
    ds = tf_data_pipeline.build_generator_pipeline(
        data_path=data_path, split='train', sort_by_length=False)

    def resample(x):
      return tf_data_pipeline.resample_example(x, max_length=5)

    # Resample each one 100 times to check bounds.
    n = 100
    ds = ds.take(1)
    ds = ds.repeat(n)
    ds = ds.map(resample, num_parallel_calls=1)
    iterator = ds.make_one_shot_iterator()
    batch = iterator.get_next()
    results = []
    with self.test_session() as s:
      try:
        for _ in range(n):
          results.append(s.run(batch))
      except tf.errors.OutOfRangeError:
        pass
    self.assertEqual(n, len(results))

    for _, x in enumerate(results):
      tokens = list(x['context_tokens'])
      start = int(x['answers_start_token'])
      end = int(x['answers_end_token'])
      self.assertEqual(['The', 'University', 'of', 'Chicago', 'also'], tokens)
      self.assertEqual(0, start)
      self.assertEqual(0, end)



if __name__ == '__main__':
  tf.test.main()
