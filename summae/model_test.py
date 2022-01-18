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

"""Tests for model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf  # tf

from summae import model as m


# pylint: disable=invalid-name


class ModelTest(tf.test.TestCase):

  def assertIntsInRange(self, i_np, a, b):
    self.assertEqual(0, np.sum(i_np >= b))
    self.assertEqual(0, np.sum(i_np < a))

  def test_prepend_token(self):
    vs = 10
    bs = 2
    es = 3
    emb_matrix_VxE = tf.random_normal([vs, es])
    x_BxLxE = tf.random.normal((bs, 10, es))
    x_prepend = m.prepend_token(x_BxLxE, emb_matrix_VxE, 2)
    self.assertEqual((bs, 11, es), x_prepend.get_shape())
    with self.session() as ss:
      [x_np, xp_np, emb_np] = ss.run([x_BxLxE, x_prepend, emb_matrix_VxE])
      # First element is special token.
      self.assertAllEqual(np.tile(emb_np[2, :], (bs, 1)),
                          np.squeeze(xp_np[:, 0, :]))
      # Rest are the same.
      self.assertAllEqual(x_np, xp_np[:, 1:, :])

  def test_gru_encoder_pooling_method(self):
    es = 7
    hs = 8
    ls = 4
    for bidirect_encode in [True, False]:
      for pooling in ['mean', 'last']:
        tf.reset_default_graph()
        encoder = m.GruEncoder(
            hidden_size=hs,
            latent_size=ls,
            scope='gru_encoder',
            bidirect_encode=bidirect_encode)
        x_BxLxE = tf.random.normal((2, 10, es))  # batch_size = 2
        seq_lengths_B = tf.constant([3, 6])
        encoded_BxH, _ = encoder.encode(x_BxLxE, seq_lengths_B, pooling, True)

        with self.session() as ss:
          ss.run(tf.initializers.global_variables())
          encoded_BxH_np = ss.run(encoded_BxH)

        self.assertEqual((2, ls), encoded_BxH_np.shape)

  def test_transformer_encoder_pooling_method(self):
    es = 8
    nl = 2
    nh = 4
    fs = 7
    hs = 8
    ls = 4
    for pooling in ['mean', 'first']:
      tf.reset_default_graph()
      encoder = m.TransformerEncoder(
          num_layers=nl,
          num_heads=nh,
          hidden_size=hs,
          filter_size=fs,
          attention_dropout=0.1,
          relu_dropout=0.1,
          postprocess_dropout=0.1,
          latent_size=4,
          scope='trf_encoder')

      x_BxLxE = tf.random.normal((2, 10, es))  # batch_size = 2
      seq_lengths_B = tf.constant([3, 6])
      encoded_BxH, _ = encoder.encode(x_BxLxE, seq_lengths_B, pooling, True)

      with self.session() as ss:
        ss.run(tf.initializers.global_variables())
        encoded_BxH_np = ss.run(encoded_BxH)

      self.assertEqual((2, ls), encoded_BxH_np.shape)

  def test_gru_decoder_teacher_force(self):
    vs = 9
    h = 10
    z = 2
    bs = 5
    es = 3
    max_seq_length = 11
    nl = 6

    for cond_only_init in [True, False]:
      if cond_only_init:
        h = z
      tf.reset_default_graph()
      emb_matrix_VxE = tf.get_variable(
          'emb',
          shape=[vs, es],
          initializer=tf.truncated_normal_initializer(stddev=0.01))
      decoder = m.GruDecoder(
          h, vs, emb_matrix_VxE, 10, nl, cond_only_init=cond_only_init)
      # Test teacher_force shapes
      seq_lengths = tf.constant([3, 2, 4, 1, 2])
      dec_out = decoder.teacher_force(
          tf.random.normal((bs, h)),  # state
          tf.random.normal((bs, max_seq_length, es)),  # dec_inputs
          seq_lengths)  # lengths

      with self.session() as ss:
        ss.run(tf.initializers.global_variables())
        dec_out_np = ss.run(dec_out)

      self.assertEqual((bs, max_seq_length, vs), dec_out_np.shape)
      # TODO(peterjliu): Check zeros beyond sequence length

  def test_gru_decoder_decode_v(self):
    vs = 9
    h = 10
    z = 2
    bs = 5
    es = 3
    nl = 6
    k = 3
    alpha = 0.6

    for cond_only_init in [True, False]:
      if cond_only_init:
        h = z
      tf.reset_default_graph()
      emb_matrix_VxE = tf.get_variable(
          'emb',
          shape=[vs, es],
          initializer=tf.truncated_normal_initializer(stddev=0.01))
      decoder = m.GruDecoder(
          h, vs, emb_matrix_VxE, 10, nl, cond_only_init=cond_only_init)

      symb = decoder.decode_v(tf.random.normal((bs, h)), method='argmax')
      symbr = decoder.decode_v(tf.random.normal((bs, h)), method='random')
      symbb = decoder.decode_v(tf.random.normal((bs, h)), method='beam',
                               first_token=0, beam_size=k, alpha=alpha)

      with self.session() as ss:
        ss.run(tf.initializers.global_variables())
        symb_np, symbr_np, symbb_np = ss.run([symb, symbr, symbb])

      self.assertEqual(bs, symb_np.shape[0])
      self.assertEqual(bs, symbr_np.shape[0])
      self.assertEqual(bs, symbb_np.shape[0])

      # Check symbols are within vocab, i.e. all following statemes are false.
      self.assertIntsInRange(symb_np, 0, vs)
      self.assertIntsInRange(symbr_np, 0, vs)
      self.assertIntsInRange(symbb_np, 0, vs)

    # TODO(peterjliu): Test _not_finished separately

  def test_gru_decoder_decode_v_gumbel(self):
    vs = 9
    h = 10
    z = 2
    bs = 5
    es = 3
    nl = 6

    for cond_only_init in [True, False]:
      if cond_only_init:
        h = z
      tf.reset_default_graph()
      emb_matrix_VxE = tf.get_variable(
          'emb',
          shape=[vs, es],
          initializer=tf.truncated_normal_initializer(stddev=0.01))
      decoder = m.GruDecoder(
          h, vs, emb_matrix_VxE, 10, nl, cond_only_init=cond_only_init)

      symb_BxM, symb_emb_BxMxE = decoder.decode_v_gumbel(
          tf.random.normal((bs, h)))

      with self.session() as ss:
        ss.run(tf.initializers.global_variables())
        symb_np, symbe_np = ss.run([symb_BxM, symb_emb_BxMxE])

      # pylint: disable=g-generic-assert
      self.assertEqual(2, len(symb_np.shape))
      self.assertEqual(3, len(symbe_np.shape))
      self.assertEqual(bs, symb_np.shape[0])
      self.assertEqual(bs, symbe_np.shape[0])
      self.assertEqual(es, symbe_np.shape[2])

      # Check symbols are within vocab, i.e. all following stateme are false.
      self.assertIntsInRange(symb_np, 0, vs)

  def test_transformer_decoder_teacher_force(self):
    vs = 9
    h = 4
    f = 12
    z = 8
    es = 4
    bs = 5
    nl = 4
    nh = 2
    max_seq_length = 11

    for cond_by_addition in [True, False]:
      if cond_by_addition:
        z = es
      tf.reset_default_graph()
      emb_matrix_VxE = tf.get_variable(
          'emb',
          shape=[vs, es],
          initializer=tf.truncated_normal_initializer(stddev=0.01))
      decoder = m.TransformerDecoder(num_layers=nl, num_heads=nh, hidden_size=h,
                                     filter_size=f, attention_dropout=0.1,
                                     relu_dropout=0.1, postprocess_dropout=0.1,
                                     embed_VxE=emb_matrix_VxE, vocab_size=vs,
                                     max_steps=10, latent_size=z,
                                     tie_embeddings=False,
                                     cond_by_addition=cond_by_addition)
      # Test teacher_force shapes
      dec_out = decoder.teacher_force(
          cond_input_BxZ=tf.random.normal((bs, z)),
          dec_inputs_BxSxE=tf.random.normal((bs, max_seq_length, es)))

      with self.session() as ss:
        ss.run(tf.initializers.global_variables())
        dec_out_np = ss.run(dec_out)
      self.assertEqual((bs, max_seq_length, vs), dec_out_np.shape)

  def test_transformer_decoder_decode_v(self):
    vs = 9
    h = 4
    f = 12
    z = 8
    es = 4
    bs = 5
    nl = 4
    nh = 2
    k = 3
    alpha = 0.6

    for cond_by_addition in [True, False]:
      if cond_by_addition:
        z = es
      tf.reset_default_graph()
      emb_matrix_VxE = tf.get_variable(
          'emb',
          shape=[vs, es],
          initializer=tf.truncated_normal_initializer(stddev=0.01))
      decoder = m.TransformerDecoder(num_layers=nl, num_heads=nh, hidden_size=h,
                                     filter_size=f, attention_dropout=0.1,
                                     relu_dropout=0.1, postprocess_dropout=0.1,
                                     embed_VxE=emb_matrix_VxE, vocab_size=vs,
                                     max_steps=10, latent_size=z,
                                     tie_embeddings=False,
                                     cond_by_addition=cond_by_addition)
      symb = decoder.decode_v(tf.random.normal((bs, z)))
      symbb = decoder.decode_v(tf.random.normal((bs, z)), method='beam',
                               first_token=0, beam_size=k, alpha=alpha)

      with self.session() as ss:
        ss.run(tf.initializers.global_variables())
        symb_np, symbb_np = ss.run([symb, symbb])

      # Check shape
      self.assertEqual(bs, symb_np.shape[0])
      self.assertEqual(bs, symbb_np.shape[0])

      # Check decoded symbols are legit IDs
      self.assertIntsInRange(symb_np, 0, vs)
      self.assertIntsInRange(symbb_np, 0, vs)

  def test_id_seq_length(self):
    self.assertAllEqual([3, 2],
                        m.id_seq_length(tf.constant([[1, 2, 3, 0, 0, 0],
                                                     [2, 3, 0, 0, 0, 0]],
                                                    dtype=tf.int64)))

  def test_create_perm_label_table(self):
    N = 3
    perm_label_table = m.create_perm_label_table(N)
    perms_B = tf.constant(['012', '021', '102', '120', '201', '210'],
                          dtype=tf.string)
    with self.session() as ss:
      ss.run(tf.tables_initializer())
      labels_B = ss.run(perm_label_table.lookup(perms_B))
    # Check if it's unique class ID for each permutation.
    self.assertAllEqual(np.sort(labels_B), np.array([0, 1, 2, 3, 4, 5]))

  def test_convert_sents_to_paragraphs(self):
    s_ids_BxNxL = tf.constant([
        [[5, 7, 1, 0], [4, 3, 1, 0], [6, 1, 0, 0]],
        [[2, 2, 3, 1], [3, 1, 0, 0], [4, 2, 1, 0]],
        [[5, 5, 5, 1], [5, 5, 5, 1], [5, 5, 1, 0]]], dtype=tf.int64)
    p_ids_BxS = m.convert_sents_to_paragraph(s_ids_BxNxL, 3)
    self.assertAllEqual(
        p_ids_BxS,
        tf.constant([[5, 7, 4, 3, 6, 1, 0, 0, 0],
                     [2, 2, 3, 3, 4, 2, 1, 0, 0],
                     [5, 5, 5, 5, 5, 5, 5, 5, 1]], dtype=tf.int64))

  def test_swap_sentences_with_scheme(self):
    # (B, N, L) = (2, 3, 6)
    s_ids_BxNxL = tf.constant([
        [[3, 2, 3, 1, 0, 0], [2, 2, 6, 3, 1, 0], [2, 3, 4, 5, 6, 1]],
        [[4, 3, 1, 0, 0, 0], [7, 6, 8, 2, 4, 1], [2, 5, 7, 7, 1, 0]]],
                              dtype=tf.int64)
    p_swapped_ids_BxS = m.corrupt_paragraph_with_scheme(s_ids_BxNxL,
                                                        scheme='last_two')
    self.assertAllEqual(
        p_swapped_ids_BxS,
        tf.constant([[3, 2, 3, 2, 3, 4, 5, 6, 2, 2, 6, 3, 1],
                     [4, 3, 2, 5, 7, 7, 7, 6, 8, 2, 4, 1, 0]],
                    dtype=tf.int64))

  def test_add_eos_2d(self):
    b = tf.constant([
        [3, 2, 4, 0,],
        [5, 9, 2, 4,],
        [5, 0, 0, 0,]], dtype=tf.int64)
    self.assertAllEqual([
        [3, 2, 4, 1, 0],
        [5, 9, 2, 4, 1],
        [5, 1, 0, 0, 0]], m.add_eos_2d(b))

  def test_random_mask_like(self):
    b = tf.constant([[3, 2, 4, 1, 0], [5, 9, 2, 4, 1], [5, 1, 0, 0, 0]],
                    dtype=tf.int64)
    l = tf.constant([4, 5, 2])
    b_mask_all_but_eos = tf.constant([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0],
                                      [1, 0, 0, 0, 0]])
    self.assertAllEqual(
        tf.zeros_like(b), m.random_mask_like(b, l, 0.0, not_mask_eos=True))
    self.assertAllEqual(b_mask_all_but_eos,
                        m.random_mask_like(b, l, 1.0, not_mask_eos=True))

  def test_mask_ids(self):
    b = tf.constant([[3, 2, 4, 1, 0], [5, 9, 2, 4, 1], [5, 1, 0, 0, 0]],
                    dtype=tf.int64)
    l = tf.constant([4, 5, 2])
    mask_id = 32583
    mask_all_but_eos = tf.constant([[mask_id, mask_id, mask_id, 1, 0],
                                    [mask_id, mask_id, mask_id, mask_id, 1],
                                    [mask_id, 1, 0, 0, 0]])

    self.assertAllEqual(b, m.mask_ids(b, l, 0.0, mask_id))
    self.assertAllEqual(mask_all_but_eos, m.mask_ids(b, l, 1.0, mask_id))

  def test_mask_embs(self):
    # suppose [1,1,1], [0,0,0], [2,2,2] are embeddings for <eos>, <pad>, <mask>
    emb_size = 3
    b_BxSxE = tf.constant([[[3, 4, 5], [3, 1, 2], [1, 1, 1], [0, 0, 0]],
                           [[2, 3, 1], [0, 1, 2], [1, 1, 1], [0, 0, 0]],
                           [[0, 3, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]],
                          dtype=tf.float32)
    b_mask_all_but_eos = tf.constant(
        [[[2, 2, 2], [2, 2, 2], [1, 1, 1], [0, 0, 0]],
         [[2, 2, 2], [2, 2, 2], [1, 1, 1], [0, 0, 0]],
         [[2, 2, 2], [1, 1, 1], [0, 0, 0], [0, 0, 0]]],
        dtype=tf.float32)
    l_B = tf.constant([3, 3, 2])
    mask_emb_E = tf.ones([emb_size]) * 2.0
    masked_BxSxE = m.mask_embs(b_BxSxE, l_B, 0.0, mask_emb_E)
    self.assertAllEqual(b_BxSxE, masked_BxSxE)
    masked_BxSxE = m.mask_embs(b_BxSxE, l_B, 1.0, mask_emb_E)
    self.assertAllEqual(b_mask_all_but_eos, masked_BxSxE)

  def test_apply_mask_to_embs(self):
    emb_size = 3
    mask_emb_E = tf.ones([emb_size]) * 2.0
    b_BxSxE = tf.constant([[[3, 4, 5], [3, 1, 2], [1, 1, 1], [0, 0, 0]],
                           [[2, 3, 1], [0, 1, 2], [1, 1, 1], [0, 0, 0]],
                           [[0, 3, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]],
                          dtype=tf.float32)
    mask_BxSx1 = tf.expand_dims(
        tf.constant([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                    dtype=tf.float32),
        axis=2)
    b_mask_true = tf.constant([[[3, 4, 5], [2, 2, 2], [1, 1, 1], [0, 0, 0]],
                               [[2, 2, 2], [0, 1, 2], [1, 1, 1], [0, 0, 0]],
                               [[0, 3, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]],
                              dtype=tf.float32)
    b_mask_BxSxE = m.apply_mask_to_embs(b_BxSxE, mask_BxSx1, mask_emb_E)
    self.assertAllEqual(b_mask_true, b_mask_BxSxE)

  def test_mask_random_pos(self):
    b = tf.constant([[3, 2, 4, 0, 0], [5, 9, 2, 4, 5], [5, 0, 0, 0, 0]],
                    dtype=tf.int64)
    keep_rate = 1.0
    self.assertAllEqual(tf.ones_like(b), m.mask_random_pos(b, keep_rate))

  def test_reduce_mean_weighted(self):
    b_N = tf.constant([0.5, 0.5])  # assume sum(b_N) = 1
    x_BxNxZ = tf.constant([[[3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6]]],
                          tf.float32)
    self.assertAllEqual(
        tf.reduce_mean(x_BxNxZ, axis=1), m.reduce_mean_weighted(x_BxNxZ, b_N))

    b_N = tf.constant([0.0, 1.0])
    self.assertAllEqual(x_BxNxZ[:, 1, :], m.reduce_mean_weighted(x_BxNxZ, b_N))

  def test_get_features_labels(self):
    B = 3
    Z = 5
    pos_np_BxZ = np.random.random((B, Z))
    neg_np_BxZ = np.random.random((B, Z))
    pos_BxZ = tf.constant(pos_np_BxZ)
    neg_BxZ = tf.constant(neg_np_BxZ)
    features_2BxZ, labels_2B = m.get_features_labels(pos_BxZ, neg_BxZ)
    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      f_np, l_np = ss.run([features_2BxZ, labels_2B])
    self.assertAllEqual(f_np, np.concatenate([pos_np_BxZ, neg_np_BxZ], axis=0))
    self.assertAllEqual(l_np, np.concatenate([np.ones(B), np.zeros(B)]))

  def test_get_discriminator(self):
    x = tf.random.normal((10, 4))
    disc_fn = m.get_discriminator(3, 'd_')
    logits_B = disc_fn(x)
    self.assertNotEmpty(tf.global_variables('d_'))
    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      l_np = ss.run(logits_B)
      self.assertEqual((10,), l_np.shape)

  def test_dense_layer_with_proj(self):
    H = 5
    O = 6
    B = 4
    S = 3
    I = 2
    x = tf.random.normal((B, S, I))
    p = tf.random.normal((H, O))
    _ = m.dense_layer_with_proj(x, p, scope='ts_')
    tf.logging.info(tf.global_variables('ts_'))
    num_vars = len(tf.global_variables('ts_'))
    _ = m.dense_layer_with_proj(x, p, scope='ts_')
    # Test that dense layer is re-used.
    self.assertLen(tf.global_variables('ts_'), num_vars)


if __name__ == '__main__':
  tf.test.main()
