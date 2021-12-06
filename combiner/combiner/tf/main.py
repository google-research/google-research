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

# pylint: skip-file
from absl import flags
from absl import app
import numpy as np
import tensorflow.compat.v1 as tf
from combiner.tf import configs
from combiner.tf import models


flags.DEFINE_string(
  'hparam_set', None,
  'String that specifies the set of params to use.')
FLAGS = flags.FLAGS


def main(argv):
  del argv
  config = getattr(configs, FLAGS.hparam_set).cfg
  batch_size = 2
  seq_len = 8
  config.model.max_seq_len = seq_len
  config.model.max_seg_len = 4
  config.model.dropout = 0.0
  config.model.dropatt = 0.0
  config.model.vocab_size = 10
  config.tpu.use_bfloat16 = False
  config.model.dtype = tf.float32
  p_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_len])
  p_outputs = tf.placeholder(tf.int32, shape=[batch_size, seq_len])
  train_tf_out = models.transformer(p_inputs, config.model, True,
                                    causal=True)
  train_logits = models.lm_head(train_tf_out['output'], config.model,
                                train_tf_out['word_embeddings'])

  nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=p_outputs, logits=train_logits)
  loss = tf.reduce_mean(nll)
  optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
  train_op = optimizer.minimize(loss)
  x = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                [0, 8, 7, 6, 5, 4, 3, 2]], dtype=np.int32)
  y = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1]], dtype=np.int32)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
      cur_loss, _ = sess.run([loss, train_op],
                             feed_dict={p_inputs: x, p_outputs: y})
      print(cur_loss)
      cur_x = np.zeros(x.shape, dtype=np.int32)
      cur_x[:, 1] = y[:, 0]
      list_pred = [y[:, 0]]
      for i in range(1, seq_len):
        cur_pred = sess.run(train_logits, feed_dict={p_inputs: cur_x})
        cur_pred = np.argmax(cur_pred[:, i], axis=-1).astype(np.int32)
        if i + 1 < seq_len:
          cur_x[:, i + 1] = cur_pred
        list_pred.append(cur_pred)
      pred = np.array(list_pred).T
      print(pred)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  app.run(main)
