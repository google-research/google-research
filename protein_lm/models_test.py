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

"""Tests for models."""

import functools
import pathlib

import gin
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow.compat.v1 as tf

from protein_lm import data
from protein_lm import domains
from protein_lm import evaluation
from protein_lm import models
from protein_lm import modules

tf.enable_eager_execution()

lm_cfg = dict(
    batch_size=4, num_layers=1, num_heads=2, emb_dim=8, mlp_dim=8, qkv_dim=8)
lm_cls = functools.partial(models.FlaxLM, **lm_cfg)


def _test_domain():
  return domains.FixedLengthDiscreteDomain(length=3, vocab_size=4)


class FlaxRegressionTest(tf.test.TestCase):

  def setUp(self):
    cls = functools.partial(
        models.FlaxModel,
        pmap=False,
        with_bos=True,
        output_head=('logits', 'regression'),
        **lm_cfg)
    self._domain = domains.FixedLengthDiscreteDomain(length=6, vocab_size=4)
    lm = cls(domain=self._domain)
    self.lm = lm
    super().setUp()

  def test_fit(self):
    bos = self.lm.bos_token
    xs = np.array([
        [bos, 0, 1, 0, 1, 0, 1],
        [bos, 1, 0, 1, 0, 1, 0],
    ])
    inputs = xs[:, :-1]
    targets = xs[:, 1:]
    reg_targets = np.cumsum(targets, axis=-1)[:, :-1]
    reg_targets = jnp.pad(
        reg_targets, [[0, 0], [1, 0]],
        mode='constant',
        constant_values=jnp.asarray(0, dtype=jnp.float32))

    batch = dict(
        inputs=inputs,
        targets=dict(classification=targets, regression=reg_targets),
        weights=dict(
            classification=np.ones_like(targets),
            regression=np.ones_like(reg_targets)))

    outputs = self.lm.preprocess(batch, mode=models.Mode.train, rng=None)
    print(outputs)

    metrics = self.lm.fit_batch(batch)
    print(metrics)
    metrics = self.lm.fit_batch(batch)
    print(metrics)


class FlaxModelTaggingTest(tf.test.TestCase):

  def test_tag_attention(self):
    # TODO(ddohan): Consider making decorator which tracks tensor distributions.
    def tagging_dot_product_attention(query, key, value, **kwargs):
      modules.Tag(jnp.mean(query), name='mean_query')
      modules.Tag(jnp.mean(query), name='mean_key')
      modules.Tag(jnp.mean(query), name='mean_value')
      return modules.nn.attention.dot_product_attention(query, key, value,
                                                        **kwargs)

    domain = _test_domain()
    lm = models.FlaxModel(
        domain=domain, attention_fn=tagging_dot_product_attention, **lm_cfg)
    xs = domain.sample_uniformly(4)
    metrics = []
    for _ in range(2):
      step_metrics = lm.fit_batch((xs, xs))
      metrics.append(step_metrics)

    combined_metrics = evaluation.combine_metrics(metrics)

    # Confirm metrics are included.
    self.assertIn('nn/1/1/mean_query', combined_metrics)
    self.assertIn('nn/1/1/mean_key', combined_metrics)
    self.assertIn('nn/1/1/mean_value', combined_metrics)

    # Check they are averaged rather than normalized by denominator.
    # key = 'nn/1/1/mean_value'
    # avg = (metrics[0][key] + metrics[1][key]) / 2.0
    # self.assertAlmostEqual(avg, combined_metrics[key])


class FlaxModelBaseTest(tf.test.TestCase):

  def setUp(self):
    cls = functools.partial(models.FlaxModel, **lm_cfg)
    self._domain = _test_domain()
    lm = cls(domain=self._domain)
    self.lm = lm
    self.xs = np.array([
        [1, 1, 0],
        [1, 1, 1],
    ])
    super().setUp()

  def test_fit(self):
    self.lm.fit((self.xs, self.xs), epochs=4, batch_size=2)

  def test_score(self):
    scores = self.lm.score(self.xs)
    self.assertEqual((2, 3, self._domain.vocab_size), scores.shape)

  def test_evaluate(self):
    metrics = self.lm.evaluate_batch((self.xs, self.xs))
    self.assertIn('accuracy', metrics)
    self.assertEqual(jnp.sum(jnp.ones_like(self.xs)), metrics['denominator'])

    metrics = self.lm.evaluate_batch(
        (self.xs, self.xs, jnp.zeros_like(self.xs)))
    self.assertEqual(0, metrics['denominator'])


class BERTTest(tf.test.TestCase):

  def setUp(self):
    cls = functools.partial(models.FlaxBERT, **lm_cfg)
    self._domain = domains.VariableLengthDiscreteDomain(
        vocab=domains.ProteinVocab(
            include_anomalous_amino_acids=True,
            include_bos=True,
            include_eos=True,
            include_pad=True,
            include_mask=True),
        length=3)

    lm = cls(domain=self._domain, grad_clip=1.0)
    self.lm = lm
    self.xs = np.array([
        [1, 1, 0],
    ])
    super().setUp()

  def test_fit(self):
    self.lm.fit(self.xs, epochs=4, batch_size=2)

  def test_score(self):
    scores = self.lm.score(self.xs)
    self.assertEqual((1, 3, self._domain.vocab_size), scores.shape)

  def test_sample(self):
    mask = self._domain.vocab.mask
    xs = np.array([
        [mask, 1, 0],
        [1, mask, 1],
        [1, 1, mask],
    ])
    rng = jrandom.PRNGKey(0)
    samples = self.lm.sample(xs, rng=rng)
    self.assertAllEqual(xs.shape, samples.shape)

    # Check masked positions are filled in.
    self.assertNotEqual(samples[0, 0], mask)
    self.assertNotEqual(samples[1, 1], mask)
    self.assertNotEqual(samples[2, 2], mask)
    unmasked = xs != mask
    # Unmasked positions are the same.
    self.assertAllEqual(xs[unmasked], samples[unmasked])

  def test_evaluate(self):
    metrics = self.lm.evaluate_batch(self.xs)
    self.assertIn('accuracy', metrics)


class BERTMaskingTest(tf.test.TestCase):

  def setUp(self):
    self._domain = data.protein_domain
    v = data.protein_domain.vocab
    pad = v.pad
    print(v.pad)
    self._xs = jnp.array([[0, 0, 0, 0, 0, 0, pad, pad],
                          [0, 0, 0, 0, 0, pad, pad, pad]])
    super().setUp()

  def test_all_mask(self):
    """Test masking with MASK values."""
    xs = self._xs
    v = self._domain.vocab
    masker = models.BertMasker(
        self._domain,
        mask_rate=1.0,
        mask_token_proportion=1.0,
        random_token_proportion=0.0)
    for k in range(10):
      rng = jrandom.PRNGKey(k)
      inputs, outputs, weights = masker(xs, rng=rng, mode=models.Mode.train)
      self.assertAllEqual((xs == v.pad), (inputs == v.pad))
      self.assertAllEqual((xs != v.pad), (inputs == v.mask))
      self.assertAllEqual(xs != v.pad, weights)
      self.assertAllEqual(xs, outputs)

  def test_all_normal(self):
    """Test masking with random values."""
    xs = self._xs
    v = self._domain.vocab
    # Check
    masker = models.BertMasker(
        self._domain,
        mask_rate=1.0,
        mask_token_proportion=0.0,
        random_token_proportion=1.0)
    for k in range(10):
      rng = jrandom.PRNGKey(k)
      inputs, outputs, weights = masker(xs, rng=rng, mode=models.Mode.train)
      is_normal = np.isin(inputs, masker._normal_tokens)
      self.assertAllEqual((xs == v.pad), (inputs == v.pad))
      self.assertAllEqual((xs != v.pad), is_normal)
      self.assertAllEqual(xs != v.pad, weights)
      self.assertAllEqual(xs, outputs)

  def test_all_identity(self):
    """Test no-mask case (maintaining original values)."""
    xs = self._xs
    v = self._domain.vocab
    masker = models.BertMasker(
        self._domain,
        mask_rate=1.0,
        mask_token_proportion=0.0,
        random_token_proportion=0.0)
    for k in range(10):
      rng = jrandom.PRNGKey(k)
      inputs, outputs, weights = masker(xs, rng=rng, mode=models.Mode.train)
      self.assertAllEqual(xs, inputs)
      self.assertAllEqual(xs != v.pad, weights)
      self.assertAllEqual(xs, outputs)


class UtilTest(tf.test.TestCase):

  def test_lr_schedule(self):
    """Test passing learning rate function as learning_rate."""
    cls = functools.partial(models.FlaxLM, **lm_cfg)

    lm = cls(
        domain=_test_domain(),
        learning_rate=models.utils.create_learning_rate_scheduler(),
        grad_clip=1.0)
    xs = np.array([
        [1, 1, 0],
    ])
    lm.fit(xs, epochs=4, batch_size=2)


class LMTest(tf.test.TestCase):

  def setUp(self):
    cls = functools.partial(models.FlaxLM, **lm_cfg)

    lm = cls(domain=_test_domain(), grad_clip=1.0)
    self.lm = lm
    self.xs = np.array([
        [1, 1, 0],
    ])
    super().setUp()

  def test_fit(self):
    self.lm.fit(self.xs, epochs=4, batch_size=2)

  def test_sample(self):
    samples = self.lm.sample(2)
    self.assertEqual((2, 3), samples.shape)

  def test_score(self):
    scores = self.lm.score(self.xs)
    self.assertEqual((1, 3, 5), scores.shape)

  def test_evaluate(self):
    metrics = self.lm.evaluate_batch(self.xs)
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

      loaded_model = models.load_model(save_dir, model_cls=models.FlaxLM)
      self.assertAllEqual(
          lm.optimizer.target.params['embed']['embedding'],
          loaded_model.optimizer.target.params['embed']['embedding'])

  def test_save_model_kwargs(self):
    model_kwargs = dict(
        num_layers=2, num_heads=2, qkv_dim=128, mlp_dim=128, emb_dim=128)
    lm = lm_cls(_test_domain(), **model_kwargs)
    ckpt_dir = self._tmpdir
    models.save_model_kwargs(ckpt_dir, lm)

    # Check we can load successfully.
    loaded_model_kwargs = models.parse_config(ckpt_dir)
    self.assertDictEqual(model_kwargs, loaded_model_kwargs)


if __name__ == '__main__':
  tf.test.main()
