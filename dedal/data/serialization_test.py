# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Test for the serialization package.

import numpy as np
import tensorflow as tf

from dedal.data import serialization


class SerializationTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)
    np.random.seed(0)
    seq_len = 25
    self.example = {
        'seq': np.arange(seq_len),
        'name': 'michel',
        'multi': np.random.uniform(size=(seq_len, 4)),
        'ints': np.random.randint(1, 10, size=(seq_len)),
    }

  def test_sequence_coder(self):
    coder = serialization.SequenceCoder(
        specs={
            'seq': tf.float32,
            'name': tf.string,
            'multi': tf.float32,
            'ints': tf.int64,
        },
        sequence_keys=['seq', 'multi', 'ints'])

    encoded = coder.encode(self.example)
    decoded = coder.decode(encoded)
    reencoded = coder.encode(decoded)
    redecoded = coder.decode(reencoded)
    self.assertAllClose(redecoded['seq'][:, 0], self.example['seq'], atol=1e-7)
    self.assertAllClose(redecoded['seq'], decoded['seq'], atol=1e-7)
    self.assertAllClose(redecoded['multi'], self.example['multi'], atol=1e-7)
    self.assertAllClose(
        redecoded['ints'][:, 0], self.example['ints'], atol=1e-7)
    self.assertEqual(redecoded['name'], self.example['name'])

  def test_flat_coder(self):
    coder = serialization.FlatCoder(specs={
        'seq': tf.float32,
        'name': tf.string,
    })
    encoded = coder.encode(self.example)
    decoded = coder.decode(encoded)
    reencoded = coder.encode(decoded)
    redecoded = coder.decode(reencoded)
    self.assertAllClose(redecoded['seq'], self.example['seq'], atol=1e-7)
    self.assertEqual(redecoded['name'], self.example['name'])

  def test_flat_coder_not_flat(self):
    coder = serialization.FlatCoder(specs={
        'seq': tf.float32,
        'name': tf.string,
        'multi': tf.float32
    })
    with self.assertRaises(TypeError):
      coder.encode(self.example)


if __name__ == '__main__':
  tf.test.main()
