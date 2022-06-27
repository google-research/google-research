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

"""Tests for SM3 optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.utils import trainer_lib
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


from routing_transformer import sparse_transformer as sptf


class RoutingTransformerTest(tf.test.TestCase):

  def testSparseTransformer(self):
    """Test sparse transformer decode."""
    with self.cached_session() as sess:
      with tf.variable_scope("sparse_transformer", reuse=tf.AUTO_REUSE):
        hparams_set = "sparse_transformer_local"
        problem = ""
        hparams = trainer_lib.create_hparams(hparams_set, problem_name=problem)
        hparams.layer_prepostprocess_dropout = 0.
        hparams.dropout = 0.
        hparams.num_encoder_layers = 0
        hparams.num_decoder_layers = 2
        hparams.local_relative = False
        hparams.query_shape = (20,)
        hparams.memory_flange = (0,)
        hparams.max_length = 200
        sparse_transformer = sptf.SparseTransformer(hparams)
        sparse_transformer.set_mode(tf_estimator.ModeKeys.PREDICT)
        sparse_transformer.vocab_size = 50
        features = {}
        decode_step = 10
        cache = {}
        # Testing that changing target tokens beyond decode_step has no effect
        # i = 0 or less should have the next cell sum == 0
        i = -5
        targets_prefix = tf.random.stateless_uniform(
            [1, decode_step - i],
            minval=0,
            maxval=sparse_transformer.vocab_size,
            dtype=tf.dtypes.int32,
            seed=(75, 48))
        zeros = tf.zeros([1, hparams.max_length - decode_step + i],
                         dtype=tf.int32)
        features["targets"] = tf.concat([targets_prefix, zeros],
                                        axis=-1)
        output_step1 = sparse_transformer.body(features,
                                               decode_step=decode_step,
                                               cache=cache)
        features["targets"] = tf.concat([
            targets_prefix, tf.random.stateless_uniform(
                [1, hparams.max_length - decode_step + i],
                minval=0,
                maxval=sparse_transformer.vocab_size,
                dtype=tf.dtypes.int32,
                seed=(67, 89))], axis=-1)
        output_step2 = sparse_transformer.body(features,
                                               decode_step=decode_step,
                                               cache=cache)
        initializer = tf.global_variables_initializer()
        if initializer is not None:
          initializer.run()

        output1_np = sess.run(output_step1)
        output2_np = sess.run(output_step2)
        self.assertEqual(output1_np.shape, output2_np.shape)


if __name__ == "__main__":
  tf.test.main()
