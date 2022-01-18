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

"""Tests for models.dedal."""

import functools

from absl.testing import parameterized
import tensorflow as tf

from dedal import multi_task
from dedal.models import aligners
from dedal.models import dedal
from dedal.models import encoders
from dedal.models import homology
from dedal.models import nlp as nlp_layers


class FakeSequenceLogits(tf.keras.Model):
  """Turns a sequence of embedding into a unique embedding."""

  def __init__(self, output_dim = 10, activation='softmax'):
    super().__init__()
    self._dense = tf.keras.layers.Dense(output_dim, activation=activation)

  def call(self, embeddings, mask=None, training=True):
    x = tf.math.reduce_mean(embeddings, axis=1)
    return self._dense(x)


def fake_align(sim_mat, gap_open, gap_extend):
  del gap_open, gap_extend
  return tf.math.reduce_mean(sim_mat, axis=(1, 2))


class ModelTest(parameterized.TestCase, tf.test.TestCase):
  """Test the generic SequenceAligner model."""

  def setUp(self):
    super().setUp()
    self.dim = 3
    self.heads_cls = multi_task.Backbone(
        embeddings=[FakeSequenceLogits],
        alignments=[homology.UncorrectedLogits,
                    homology.LogCorrectedLogits])
    self.model = dedal.Dedal(
        encoder_cls=encoders.OneHotEncoder,
        aligner_cls=functools.partial(
            aligners.SoftAligner, gap_pen_cls=aligners.ConstantGapPenalties,
            align_fn=fake_align),
        heads_cls=self.heads_cls)

    batch = 32
    seq_len = 100
    self.inputs = tf.random.uniform(
        (batch, seq_len), maxval=35, dtype=tf.int32)

  @parameterized.parameters([
      ([], []), ([0], []), ([], [0]), ([], [0, 1])])
  def test_call(self, embed_off, align_off):
    selector = self.heads_cls.constant_copy(True)
    for offs, level in zip((embed_off, align_off), selector.levels):
      for off in offs:
        level[off] = False

    preds = self.model(self.inputs, selector=selector)
    self.assertLen(preds.embeddings, 1)
    self.assertLen(preds.alignments, 2)
    self.assertIsInstance(preds.embeddings[0], tf.Tensor)
    self.assertIsInstance(preds.alignments[0], tf.Tensor)
    for i, emb_out in enumerate(preds.embeddings):
      self.assertEqual(emb_out.shape, (0) if i in embed_off else (32, 10))
    for i, align_out in enumerate(preds.alignments):
      self.assertEqual(align_out.shape, (0) if i in align_off else (32, 1))

  def test_without_negatives(self):
    model = dedal.Dedal(
        encoder_cls=encoders.OneHotEncoder,
        aligner_cls=functools.partial(
            aligners.SoftAligner, gap_pen_cls=aligners.ConstantGapPenalties,
            align_fn=fake_align),
        heads_cls=self.heads_cls,
        process_negatives=False)
    preds = model(self.inputs)
    self.assertEqual(preds.embeddings[0].shape, (32, 10))
    self.assertEqual(preds.alignments[0].shape, (16, 1))
    self.assertEqual(preds.alignments[1].shape, (16, 1))

  def test_merge(self):
    batch = 16
    zeros9 = tf.zeros((batch, 9))
    ones9 = tf.zeros((batch, 9))
    pos = [tf.zeros((batch, 3)),
           tf.zeros((batch,)),
           (zeros9, zeros9, (zeros9, zeros9, tf.zeros((batch, 4))))]
    neg = [tf.ones((batch, 3)),
           tf.ones((batch,)),
           (ones9, ones9, (ones9, ones9, tf.ones((batch, 4))))]
    merged = dedal.merge(pos, neg)
    self.assertLen(merged, 3)
    self.assertEqual(merged[0].shape, (batch * 2, 3))
    self.assertLen(merged[2], 3)
    self.assertEqual(merged[2][1].shape, (batch * 2, 9))
    self.assertLen(merged[2][2], 3)
    self.assertEqual(merged[2][2][2].shape, (batch * 2, 4))
    self.assertEqual(merged[2][2][1].shape, (batch * 2, 9))

  def test_alignment_outputs(self):
    """A test with complex outputs."""
    heads_cls = multi_task.Backbone(
        embeddings=[FakeSequenceLogits],
        alignments=[dedal.Selector,
                    homology.UncorrectedLogits,
                    homology.LogCorrectedLogits])
    model = dedal.Dedal(
        encoder_cls=encoders.OneHotEncoder,
        aligner_cls=functools.partial(
            aligners.SoftAligner, gap_pen_cls=aligners.ConstantGapPenalties,
            align_fn=fake_align),
        heads_cls=heads_cls,
        process_negatives=True)
    preds = model(self.inputs)
    self.assertEqual(preds.embeddings[0].shape, (32, 10))
    self.assertEqual(preds.alignments[1].shape, (32, 1))
    self.assertEqual(preds.alignments[2].shape, (32, 1))
    aligment_pred = preds.alignments[0]
    self.assertLen(aligment_pred, 3)
    scores, paths, sw_params = aligment_pred
    self.assertEqual(scores.shape, (32,))
    self.assertLen(sw_params, 3)
    self.assertIsNone(paths)
    self.assertEqual(sw_params[0].shape, (32, 100, 100))

  def test_disable_backprop_embeddings(self):
    encoder_cls = functools.partial(encoders.LookupEncoder, emb_dim=32)
    aligner_cls = functools.partial(
        aligners.SoftAligner, gap_pen_cls=aligners.ConstantGapPenalties)
    heads_cls = multi_task.Backbone(
        embeddings=[nlp_layers.DensePerTokenOutputHead],
        alignments=[dedal.Selector, homology.LogCorrectedLogits])
    backprop = multi_task.Backbone(embeddings=[False], alignments=[True, True])
    model = dedal.Dedal(encoder_cls=encoder_cls,
                        aligner_cls=aligner_cls,
                        heads_cls=heads_cls,
                        backprop=backprop)
    inputs = tf.random.uniform(
        (4, 16), maxval=25, dtype=tf.int32)

    with tf.GradientTape(persistent=True) as tape:
      y_pred = model(inputs).flatten()
      dummy_loss = tf.reduce_sum(y_pred['embeddings/0'] ** 2)
    grads_encoder = tape.gradient(
        dummy_loss, model.encoder.trainable_variables[0],
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertAllClose(tf.linalg.norm(grads_encoder), 0.0)

    with tf.GradientTape(persistent=True) as tape:
      y_pred = model(inputs).flatten()
      dummy_loss = tf.reduce_sum(y_pred['alignments/1'] ** 2)
    grads_encoder = tape.gradient(
        dummy_loss, model.encoder.trainable_variables[0],
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertGreater(tf.linalg.norm(grads_encoder), 0.0)

  def test_disable_backprop_homology(self):
    encoder_cls = functools.partial(encoders.LookupEncoder, emb_dim=32)
    aligner_cls = functools.partial(
        aligners.SoftAligner, gap_pen_cls=aligners.ConstantGapPenalties)
    heads_cls = multi_task.Backbone(
        embeddings=[nlp_layers.DensePerTokenOutputHead],
        alignments=[dedal.Selector, homology.LogCorrectedLogits])
    backprop = multi_task.Backbone(embeddings=[True], alignments=[True, False])
    model = dedal.Dedal(encoder_cls=encoder_cls,
                        aligner_cls=aligner_cls,
                        heads_cls=heads_cls,
                        backprop=backprop)
    inputs = tf.random.uniform(
        (4, 16), maxval=25, dtype=tf.int32)

    with tf.GradientTape(persistent=True) as tape:
      y_pred = model(inputs).flatten()
      dummy_loss = tf.reduce_sum(y_pred['alignments/1'] ** 2)
    grads_encoder = tape.gradient(
        dummy_loss, model.encoder.trainable_variables[0],
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertAllClose(tf.linalg.norm(grads_encoder), 0.0)

    with tf.GradientTape(persistent=True) as tape:
      y_pred = model(inputs).flatten()
      dummy_loss = tf.reduce_sum(y_pred['embeddings/0'] ** 2)
    grads_encoder = tape.gradient(
        dummy_loss, model.encoder.trainable_variables[0],
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertGreater(tf.linalg.norm(grads_encoder), 0.0)


class MultiInputModelTest(parameterized.TestCase, tf.test.TestCase):
  """Test the generic SequenceAligner model in multi-input mode."""

  def setUp(self):
    super().setUp()
    self.dim = 3
    self.heads_cls = multi_task.Backbone(
        embeddings=[FakeSequenceLogits],
        alignments=[homology.UncorrectedLogits,
                    homology.LogCorrectedLogits])
    self.model = dedal.Dedal(
        encoder_cls=encoders.OneHotEncoder,
        aligner_cls=functools.partial(
            aligners.SoftAligner, gap_pen_cls=aligners.ConstantGapPenalties,
            align_fn=fake_align),
        heads_cls=self.heads_cls)

    batch1, batch2 = 32, 16
    seq_len1, seq_len2 = 100, 50
    self.inputs1 = tf.random.uniform(
        (batch1, seq_len1), maxval=35, dtype=tf.int32)
    self.inputs2 = tf.random.uniform(
        (batch2, seq_len2), maxval=35, dtype=tf.int32)

    self.switch = multi_task.SwitchBackbone(embeddings=[1], alignments=[0, 0])

  @parameterized.parameters([
      ([], []), ([0], []), ([], [0]), ([], [0, 1])])
  def test_call(self, embed_off, align_off):
    selector = self.heads_cls.constant_copy(True)
    for offs, level in zip((embed_off, align_off), selector.levels):
      for off in offs:
        level[off] = False
    outputs1 = self.model(self.inputs1, selector=selector, training=False)
    outputs2 = self.model(self.inputs2, selector=selector, training=False)

    self.model.switch = self.switch
    outputs = self.model(
        [self.inputs1, self.inputs2], selector=selector, training=False)
    self.model.switch = None

    self.assertAllClose(outputs2.embeddings[0], outputs.embeddings[0])
    self.assertAllClose(outputs1.alignments[0], outputs.alignments[0])
    self.assertAllClose(outputs1.alignments[1], outputs.alignments[1])


if __name__ == '__main__':
  tf.test.main()
