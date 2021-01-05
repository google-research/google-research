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

"""Tests for supcon.launcher."""

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from supcon import enums
from supcon import hparams
from supcon import launcher


class LauncherTest(tf.test.TestCase, parameterized.TestCase):
  num_classes = 10
  training_set_size = 20
  batch_size = 5
  num_views = 2
  embedding_size = 2048

  def make_trainer(self, mode):
    return launcher.ContrastiveTrainer(
        model_inputs=tf.random.normal(
            shape=[self.batch_size, 224, 224, 3 * self.num_views],
            mean=0,
            stddev=.1,
            dtype=tf.float32),
        labels=tf.random.uniform(
            shape=[self.batch_size],
            minval=0,
            maxval=self.num_classes,
            dtype=tf.int32),
        train_global_batch_size=self.batch_size,
        hparams=hparams.HParams(
            bs=self.batch_size, eval=hparams.Eval(batch_size=self.batch_size)),
        mode=mode,
        num_classes=self.num_classes,
        training_set_size=self.training_set_size,
        is_tpu=False)

  def test_train(self):
    trainer = self.make_trainer(enums.ModelMode.TRAIN)
    scaffold = trainer.scaffold_fn()()
    train_op = trainer.train_op()
    with self.cached_session() as sess:
      sess.run(tf.initializers.global_variables())
      scaffold.init_fn(sess)
      sess.run(train_op)

  def test_eval(self):
    trainer = self.make_trainer(enums.ModelMode.EVAL)
    metric_fn, metrics = trainer.eval_metrics()
    results_dict = metric_fn(**metrics)
    with self.cached_session() as sess:
      sess.run(tf.initializers.global_variables())
      sess.run(tf.initializers.local_variables())
      sess.run(results_dict)

  @parameterized.parameters('contrastive_train', 'contrastive_eval')
  def test_inference(self, key):
    self.num_views = 1
    trainer = self.make_trainer(enums.ModelMode.INFERENCE)
    signature_def_map = trainer.signature_def_map()
    sig_def = signature_def_map[key]
    self.assertEqual([self.batch_size, self.embedding_size],
                     sig_def['embeddings'].shape.as_list())
    self.assertEqual(trainer.model_inputs.dtype, sig_def['embeddings'].dtype)
    self.assertEqual([self.batch_size, self.embedding_size],
                     sig_def['unnormalized_embeddings'].shape.as_list())
    self.assertEqual(trainer.model_inputs.dtype,
                     sig_def['unnormalized_embeddings'].dtype)
    self.assertEqual([
        self.batch_size, trainer.hparams.architecture.projection_head_layers[-1]
    ], sig_def['projection'].shape.as_list())
    self.assertEqual(trainer.model_inputs.dtype, sig_def['projection'].dtype)
    self.assertEqual([self.batch_size, self.num_classes],
                     sig_def['logits'].shape.as_list())
    self.assertEqual(trainer.model_inputs.dtype, sig_def['logits'].dtype)


if __name__ == '__main__':
  tf.test.main()
