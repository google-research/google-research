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

# Lint as: python3
"""Tests for models."""

import functools
import pathlib
import gin
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v1 as tf

from protein_lm import domains
from protein_lm import models

tf.enable_eager_execution()

lm_cfg = dict(
    batch_size=1, num_layers=2, num_heads=2, emb_dim=32, mlp_dim=32, qkv_dim=32)
lm_cls = functools.partial(models.FlaxLM, **lm_cfg)


def _test_domain():
  return domains.FixedLengthDiscreteDomain(length=3, vocab_size=4)


class TransformerTest(tf.test.TestCase):

  def test_runs(self):
    xs = np.array([
        [1, 1, 0],
    ])
    lm = lm_cls(domain=_test_domain(), grad_clip=1.0)
    lm.fit(xs, epochs=4, batch_size=2)
    samples = lm.sample(2)

    self.assertEqual((2, 3), samples.shape)
    scores = lm.score(xs)
    self.assertEqual((1, 3, 5), scores.shape)
    metrics = lm.evaluate_batch(xs)
    self.assertIn('accuracy', metrics)

  def test_overfit(self):
    domain = domains.VariableLengthDiscreteDomain(
        vocab=domains.Vocabulary(
            tokens=['a', 'b', 'c'], include_bos=True, include_eos=True),
        length=9)
    seqs = [
        list('abcabcab'),
        list('bbbbbb'),
        list('cbacbacb'),
    ]
    enc = domain.encode(seqs, pad=True)
    self.assertAllEqual(
        [[0, 1, 2, 0, 1, 2, 0, 1, 4],
         [1, 1, 1, 1, 1, 1, 4, 4, 4],
         [2, 1, 0, 2, 1, 0, 2, 1, 4]
         ], enc)
    enc = np.array(enc)
    model = lm_cls(
        domain=domain,
        learning_rate=0.01,
        dropout_rate=0.0,
        attention_dropout_rate=0.0)
    for _ in range(100):
      metrics = model.fit_batch(enc)

    # 2 less than perfect because the first token is unpredictable given just
    # <BOS>, and there are 3 total examples.
    denom = metrics['denominator'][0]
    correct = metrics['accuracy'][0]
    self.assertEqual((denom - 2)/denom, correct / denom)

  def test_bos_does_not_appear_in_fixed_len_output(self):
    """Tests that BOS is overridden in fixed length domain samples."""
    domain = domains.FixedLengthDiscreteDomain(vocab_size=2, length=10)
    lm = lm_cls(domain=domain)
    samples = lm.sample(10)
    for sample in samples:
      self.assertNotIn(lm.bos_token, sample)

  def test_bos_does_not_appear_in_var_len_output(self):
    """Tests that BOS is not used for padding in var-len domain samples."""
    domain = domains.VariableLengthDiscreteDomain(
        vocab=domains.Vocabulary(tokens=[0, 1], include_eos=True),
        length=10,
    )
    lm = lm_cls(domain=domain)
    samples = lm.sample(10)
    for sample in samples:
      self.assertNotIn(lm.bos_token, sample)

  def test_only_eos_after_eos(self):
    """Tests that the characters found after EOS are all equal to EOS."""
    domain = domains.VariableLengthDiscreteDomain(
        vocab=domains.Vocabulary(tokens=[0, 1], include_eos=True),
        length=10,
    )
    lm = lm_cls(domain=domain)
    samples = lm.sample(10)
    for sample in samples:
      if lm.eos_token in sample:
        start_eos = np.argwhere(sample == lm.eos_token)[0][0]
        self.assertAllEqual(sample[start_eos:],
                            [lm.eos_token] * (len(sample) - start_eos))

  def test_compute_logprob(self):
    domain = _test_domain()
    lm = lm_cls(domain=domain)
    seq = domain.sample_uniformly(16, seed=0)
    metrics = lm.evaluate_batch(seq)
    log_likelihoods = models.compute_logprob(seq, lm)
    # evaluate_batch() returns an array of per-batch metrics
    # for each device. Here, we use [0] because the tests run using only
    # one device.
    self.assertAllClose(metrics['loss'][0], -jnp.sum(log_likelihoods))


class CheckpointTest(tf.test.TestCase):

  def setUp(self):
    self._tmpdir = pathlib.PurePath(self.create_tempdir().full_path)
    super().setUp()

  def test_save_and_load_checkpoint_succeeds(self):
    lm = lm_cls(_test_domain())

    save_dir = self._tmpdir / 'save_ckpt'
    lm.save_checkpoint(save_dir)
    original_optimizer = lm.optimizer

    # Check we can load successfully.
    lm.load_checkpoint(save_dir)
    self.assertIsNot(lm.optimizer, original_optimizer)

  def test_load_empty_checkpoint_fails(self):
    # Check attempting to load raises an exception when there is no checkpoint.
    lm = lm_cls(domain=_test_domain())
    original_optimizer = lm.optimizer
    with self.assertRaises(ValueError):
      lm.load_checkpoint(self._tmpdir / 'notreal')
    self.assertIs(original_optimizer, lm.optimizer)

  def test_load_model(self):
    with gin.config_scope('test'):
      for k, v in lm_cfg.items():
        gin.bind_parameter('FlaxLM.%s' % k, v)

      lm = models.FlaxLM(domain=_test_domain(), random_seed=1)

      save_dir = self._tmpdir / 'save_ckpt'
      lm.save_checkpoint(save_dir)
      config_str = gin.operative_config_str()
      with tf.gfile.GFile(str(save_dir / 'config.gin'), 'w') as f:
        f.write(config_str)

      loaded_model = models.load_model(save_dir)
      self.assertAllEqual(
          lm.optimizer.target.params['embed']['embedding'],
          loaded_model.optimizer.target.params['embed']['embedding'])


if __name__ == '__main__':
  tf.test.main()
