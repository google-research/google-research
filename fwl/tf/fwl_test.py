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

import itertools
from absl.testing import absltest
import fwl
import tensorflow as tf


class FWLTest(absltest.TestCase):
  def test_fwl(self):
    # create some random inputs and pick a token to check the FWL output on
    batch_size, seq_len, fwl_size, vocab_size = 2, 8, 3, 5
    i, j = 1, 5
    x = tf.random.uniform([batch_size, seq_len, fwl_size])
    labels = tf.one_hot(tf.random.uniform(
        [batch_size, seq_len], maxval=vocab_size, dtype=tf.int32),
                        vocab_size, dtype=tf.float32)
    weights = tf.cast(tf.random.uniform([batch_size, seq_len],
                                        maxval=2, dtype=tf.int32), tf.float32)

    block = fwl.FWBlock(4, 5, 2)
    logits = block(x, labels, weights)

    # update FWL params based on the losses of previous tokens then re-run
    # the slow weight pass; this should match the FWBlock's output
    with tf.GradientTape(persistent=True) as tape:
      sw_logits = block.fwd(x, False)
      log_probs = -tf.reduce_sum(tf.nn.log_softmax(sw_logits) * labels, axis=-1)
      losses = weights * log_probs / tf.maximum(tf.reduce_sum(weights), 1e-8)
      loss_ij = tf.reduce_sum(losses[i][:j])

    all_params = [l.variables for l in block.layers[:-1]]
    all_params = list(itertools.chain(*all_params))
    grads = tape.gradient(loss_ij, all_params)
    for grad, param in zip(grads, all_params):
      if grad is not None:
        param.assign_add(-0.01 * grad)

    fw_logits = block.fwd(x, False)
    tf.debugging.assert_near(fw_logits[i, j], logits[i, j])


if __name__ == "__main__":
  absltest.main()
