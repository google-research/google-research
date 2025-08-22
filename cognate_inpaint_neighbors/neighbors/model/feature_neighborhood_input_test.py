# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""FeatureNeighborhood input tests."""

import os

import feature_neighborhood_input as fn

from lingvo import compat as tf
from lingvo.core import test_utils


class FeatureNeighborhoodInputTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    testdata_base = os.path.join(tf.flags.FLAGS.test_srcdir,
                                 "neighborhood/model/testdata")
    (self._opts, self._isymbols,
     self._osymbols) = fn.FeatureNeighborhoodInput.BasicConfig(testdata_base,
                                                               "Zway.syms")
    self._tfrecords = os.path.join(testdata_base, "Zway_test.tfrecords")
    self.assertTrue(os.path.isfile(self._tfrecords))

  def testSymbolTables(self):
    self.assertEqual(self._isymbols.find("<s>"), 1)
    self.assertEqual(self._osymbols.find("<s>"), 1)

  def testFeatureInputReaderSmallBatch(self):

    p = fn.FeatureNeighborhoodInput.Params()
    p.file_pattern = "tfrecord:" + self._tfrecords
    p.file_random_seed = 42
    p.file_buffer_size = 2
    p.file_parallelism = 1
    p.num_batcher_threads = 1
    p.bucket_upper_bound = [1024]
    p.bucket_batch_limit = [2]
    p.feature_neighborhood_input = self._opts
    inp = fn.FeatureNeighborhoodInput(p)
    with self.session(use_gpu=False) as sess:
      r = sess.run(inp.GetPreprocessedInputBatch())
      tf.logging.info("r.cognate_id = %r", r.cognate_id)
      tf.logging.info("r.spelling = %r", r.spelling)
      tf.logging.info("r.pronunciation = %r", r.pronunciation)
      self.assertEqual(r.cognate_id.shape, (2,))
      self.assertEqual(r.spelling.shape, (2, 20))
      self.assertEqual(r.pronunciation.shape, (2, 40))
      self.assertEqual(r.neighbor_spellings.shape, (2, 50, 20))
      self.assertEqual(r.neighbor_pronunciations.shape, (2, 50, 40))

      # Cognate IDs.
      ref_cognate_id = [b"61-19", b"63-19"]
      self.assertAllEqual(r.cognate_id, ref_cognate_id)

      # Zway + eos, Zway + eos
      ref_spelling = [
          [16, 70, 17, 74, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [16, 70, 17, 74, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      ]
      self.assertAllEqual(r.spelling, ref_spelling)
      print(list(r.pronunciation))
      ref_pronunciation = [
          [  # d ə m </s>
              24, 79, 46, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0
          ],
          [  # ʔ a tʼ ɨ m </s>
              89, 17, 65, 81, 46, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0
          ],
      ]
      self.assertAllEqual(r.pronunciation, ref_pronunciation)

  def testFeatureInputReaderLargeBatch(self):

    p = fn.FeatureNeighborhoodInput.Params()
    p.file_pattern = "tfrecord:" + self._tfrecords
    p.file_random_seed = 42
    p.file_buffer_size = 2
    p.file_parallelism = 1
    p.num_batcher_threads = 1
    p.bucket_upper_bound = [1024]
    p.bucket_batch_limit = [32]
    p.feature_neighborhood_input = self._opts
    inp = fn.FeatureNeighborhoodInput(p)
    with self.session(use_gpu=False) as sess:
      r = sess.run(inp.GetPreprocessedInputBatch())
      self.assertEqual(r.spelling.shape, (32, 20))
      self.assertEqual(r.pronunciation.shape, (32, 40))
      self.assertEqual(r.neighbor_spellings.shape, (32, 50, 20))
      self.assertEqual(r.neighbor_pronunciations.shape, (32, 50, 40))

  def testFeatureInputReaderLargeBatchNoNeighbors(self):

    p = fn.FeatureNeighborhoodInput.Params()
    p.file_pattern = "tfrecord:" + self._tfrecords
    p.file_random_seed = 42
    p.file_buffer_size = 2
    p.file_parallelism = 1
    p.num_batcher_threads = 1
    p.bucket_upper_bound = [1024]
    p.bucket_batch_limit = [32]
    p.feature_neighborhood_input = self._opts
    p.use_neighbors = False
    inp = fn.FeatureNeighborhoodInput(p)
    with self.session(use_gpu=False) as sess:
      r = sess.run(inp.GetPreprocessedInputBatch())
      self.assertEqual(r.spelling.shape, (32, 20))
      self.assertEqual(r.pronunciation.shape, (32, 40))
      self.assertIsNone(r.Get("neighbor_spellings"))
      self.assertIsNone(r.Get("neighbor_pronunciations"))


if __name__ == "__main__":
  tf.test.main()
