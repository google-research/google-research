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

"""Tests for pors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import parameterized
from six.moves import range
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v1 as tf  # tf
from summae import pors
from summae import util

# pylint: disable=invalid-name
FLAGS = flags.FLAGS
flags.declare_key_flag('task')
flags.declare_key_flag('use_tpu')
flags.declare_key_flag('pretrain_as_autoencoder')
flags.declare_key_flag('in_domain_pretrain_steps')
flags.declare_key_flag('out_domain_pretrain_steps')
flags.declare_key_flag('decode_reconstructions')


class PorsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PorsTest, self).setUp()
    FLAGS.task = 'rocstories'
    FLAGS.use_tpu = False
    FLAGS.pretrain_as_autoencoder = True
    FLAGS.decode_reconstructions = False
    FLAGS.in_domain_pretrain_steps = 0
    FLAGS.out_domain_pretrain_steps = 0
    self.data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    tf.set_random_seed(1234)
    self.params = {
        'decode_length': 5,
        'encoder_type': 'rnn',
        'decoder_type': 'rnn',
        'latent_size': 4,
        'decoding_method': 'argmax',
        'decoding_beam_size': 2,
        'decoding_alpha': 1.0,
        'rnn_num_layers': 3,
        'rnn_hidden_size': 4,
        'rnn_pooling': 'last',
        'rnn_bidirect_encode': False,
        'trf_hidden_size': 10,
        'trf_num_layers': 3,
        'trf_num_heads': 2,
        'trf_filter_size': 7,
        'trf_postprocess_dropout': 0.1,
        'trf_attention_dropout': 0.1,
        'trf_relu_dropout': 0.1,
        'trf_pooling': 'mean',
        'pretrain_order': 'simultaneous',
        'first_pretrain_steps': 0,
        'nsp_pretrain': False,
        'nsp_pretrain_not_next_diff_paragraph_prob': 0.5,
        'cpp_pretrain_scheme': '',
        'lm_pretrain_dec': False,
        'lambda_lm_pretrain_s': 1.0,
        'lambda_lm_pretrain_p': 1.0,
        'lambda_nsp_pretrain': 1.0,
        'lambda_cpp_pretrain': 1.0,
        'lambda_p': 1.0,
        'lambda_s': 1.0,
        'lambda_c': 1.0,
        'lambda_c_avg': 1.0,
        'embedding_size': 10,
        'learning_rate': 0.01,
        'clip_norm': 10.0,
        'tie_sent_para_enc': False,
        'tie_sent_para_dec': False,
        'tie_embeddings': False,
        'max_decode_steps': 10,
        'vocab_size': 10,
        'mask_prob_input': 0.0,
        'mask_rate_input': 0.0,
        'mask_type': 'both',
        'mask_prob_summary': 0.0,
        'mask_rate_summary': 0.0,
        'gs_temp': 2.0,
        'lambda_n': 0.0,
        'd_hidden': 3,
        'gd_step_ratio': 1,
        'adv_weight': 0.5,
        'add_critic': True,
        'lambda_c_avg2': 0.1,
        'noisy_paragraph_prob': 0.0,
        'avg_first': False,
        'train_phase_subset': 'all',
        'noisy_paragraph_scheme': 'shuffle_sentences'
    }
    self.params.update(pors.flags_hypers())

  def test_pors_model_pretrain_noD(self):
    # Tests that disciminator is not updated during pre-training.
    FLAGS.in_domain_pretrain_steps = 10
    tf.reset_default_graph()
    # Batch of 2 paragraphs each with 3 sentences.
    sids = tf.constant([
        [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
        [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
    ],
                       dtype=tf.int64)

    pids = tf.constant([[3, 2, 4, 2, 4, 2, 1, 0], [5, 2, 3, 4, 2, 5, 5, 1]],
                       dtype=tf.int64)
    features = {'sentences': sids, 'paragraphs': pids}
    ret_tensors = pors.pors_model(features, self.params, True)

    # Verify that ae_vars and d_vars is all vars.
    d_vars = tf.trainable_variables('disc_')

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      # 2 steps where we only train generator
      d_vars_np1 = ss.run(d_vars)
      _, _, _, _ = ss.run(ret_tensors[:-1])
      d_vars_np2 = ss.run(d_vars)
      for i in range(len(d_vars_np1)):
        self.assertAllClose(d_vars_np1[i], d_vars_np2[i])

  @parameterized.named_parameters(
      ('autoencoder_tpu',
       True, False, False, '', True, 'simultaneous'),
      ('autoencoder_notpu',
       True, False, False, '', False, 'simultaneous'),
      ('simultaneous_tpu',
       False, True, True, 'last_two', True, 'simultaneous'),
      ('simultaneous_notpu',
       False, True, True, 'last_two', False, 'simultaneous'),
      ('enc_first_tpu',
       False, True, True, 'last_two', True, 'encoder_first'),
      ('enc_first_notpu',
       False, True, True, 'last_two', False, 'encoder_first'),
      ('dec_first_tpu',
       False, True, True, 'last_two', True, 'decoder_first'),
      ('dec_first_notpu',
       False, True, True, 'last_two', False, 'decoder_first'),
  )
  def test_pors_model_pretrain_methods(self, pretrain_as_autoencoder,
                                       lm_pretrain_dec,
                                       nsp_pretrain,
                                       cpp_pretrain_scheme,
                                       use_tpu, pretrain_order):
    FLAGS.out_domain_pretrain_steps = 1
    FLAGS.in_domain_pretrain_steps = 1
    FLAGS.pretrain_as_autoencoder = pretrain_as_autoencoder
    self.params.update({
        'add_critic': False,
        'vocab_size': 11,
        'pretrain_order': pretrain_order,
        'first_pretrain_steps': 1,
        'lm_pretrain_dec': lm_pretrain_dec,
        'nsp_pretrain': nsp_pretrain,
        'cpp_pretrain_scheme': cpp_pretrain_scheme,
        'encoder_type': 'transformer'
    })
    FLAGS.use_tpu = use_tpu
    tf.reset_default_graph()
    # Batch of 4 paragraphs each with 5 sentences.
    sids = tf.constant([
        [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0], [4, 2, 4, 1, 0],
         [2, 1, 0, 0, 0]],
        [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0], [4, 2, 5, 1, 0],
         [5, 1, 0, 0, 0]],
        [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0], [4, 2, 4, 1, 0],
         [2, 1, 0, 0, 0]],
        [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0], [4, 2, 5, 1, 0],
         [5, 1, 0, 0, 0]]], dtype=tf.int64)

    pids = tf.constant([
        [3, 2, 4, 2, 4, 2, 4, 2, 4, 2, 1, 0],
        [5, 2, 3, 4, 2, 5, 5, 4, 2, 5, 5, 1],
        [3, 2, 4, 2, 4, 2, 4, 2, 4, 2, 1, 0],
        [5, 2, 3, 4, 2, 5, 5, 4, 2, 5, 5, 1]], dtype=tf.int64)
    features = {'sentences': sids, 'paragraphs': pids}
    ret_tensors = pors.pors_model(
        features, self.params, True, spid_dict={pors.MASK: 10})

    # Verify that ae_vars and d_vars is all vars.
    ae_vars = tf.trainable_variables('ae_')
    d_vars = tf.trainable_variables('disc_')
    self.assertEqual(set(tf.trainable_variables()), set(ae_vars + d_vars))

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      ss.run(tf.tables_initializer())
      # pre-train the first component on out-domain data for 1 step
      # If simultaneous or autoencoder then encoder and decoder are jointly
      # pre-trained.
      loss, _, _, _ = ss.run(ret_tensors[:-1])
      self.assertGreater(loss, 0)
      # pre-train the second component on in-domain data for 1 step
      loss, _, _, _ = ss.run(ret_tensors[:-1])
      self.assertGreater(loss, 0)
      # 1 regular training step
      loss, _, _, _ = ss.run(ret_tensors[:-1])
      self.assertGreater(loss, 0)

  @parameterized.named_parameters(
      ('use_tpu', True),
      ('no_tpu', False),
  )
  def test_pors_model_out_domain_pretrain(self, use_tpu):
    FLAGS.out_domain_pretrain_steps = 0
    FLAGS.in_domain_pretrain_steps = 0
    FLAGS.use_tpu = use_tpu
    # Batch of 2 paragraphs each with 3 sentences.
    sids = tf.constant([
        [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
        [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
    ],
                       dtype=tf.int64)

    pids = tf.constant([[3, 2, 4, 2, 4, 2, 1, 0], [5, 2, 3, 4, 2, 5, 5, 1]],
                       dtype=tf.int64)
    features = {'sentences': sids, 'paragraphs': pids}
    ret_tensors = pors.pors_model(features, self.params, True)

    # Verify that ae_vars and d_vars is all vars.
    ae_vars = tf.trainable_variables('ae_')
    d_vars = tf.trainable_variables('disc_')
    self.assertEqual(set(tf.trainable_variables()), set(ae_vars + d_vars))

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      d_vars_np1 = ss.run(d_vars)
      # 2 steps where we only train generator
      loss, _, _, _ = ss.run(ret_tensors[:-1])
      loss, _, _, _ = ss.run(ret_tensors[:-1])
      self.assertGreater(loss, 0)
      d_vars_np2 = ss.run(d_vars)
      for i in range(len(d_vars_np1)):
        # D-vars should not update
        self.assertAllClose(d_vars_np1[i], d_vars_np2[i])
      self.assertGreater(loss, 0)

  @parameterized.named_parameters(
      ('use_tpu_critic', True, True),
      ('use_tpu_nocritic', True, False),
      ('no_tpu_critic', False, True),
      ('no_tpu_nocritic', False, False),
  )
  def test_pors_model(self, use_tpu, add_critic):
    # TODO(peterjliu): Actually test on TPU. Setting this flag is not enough.
    FLAGS.use_tpu = use_tpu
    FLAGS.decode_reconstructions = True
    self.params.update({'add_critic': add_critic})
    tf.reset_default_graph()
    # Batch of 2 paragraphs each with 3 sentences.
    sids = tf.constant([
        [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
        [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
    ],
                       dtype=tf.int64)

    pids = tf.constant([[3, 2, 4, 2, 4, 2, 1, 0], [5, 2, 3, 4, 2, 5, 5, 1]],
                       dtype=tf.int64)
    features = {'sentences': sids, 'paragraphs': pids}
    ret_tensors = pors.pors_model(features, self.params, True)

    # Verify that ae_vars and d_vars is all vars.
    ae_vars = tf.trainable_variables('ae_')
    d_vars = tf.trainable_variables('disc_')
    self.assertEqual(set(tf.trainable_variables()), set(ae_vars + d_vars))

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      # 2 steps to train both discriminator/generator
      loss, _, _, pred_dict = ss.run(ret_tensors[:-1])
      if not use_tpu:
        self.assertIn('decoded_paragraph', pred_dict, msg=str(pred_dict))
        self.assertIn('decoded_sentences', pred_dict, msg=str(pred_dict))
      self.assertGreater(loss, 0)
      loss, _, _, _ = ss.run(ret_tensors[:-1])
      self.assertGreater(loss, 0)
      # This fails for some reason on TPU. TODO(peterjliu): Figure out why.
      # loss, _, _, _ = ss.run(ret_tensors[:-1])

  @parameterized.named_parameters(
      ('rnn_rnn_tpu', 'rnn', 'rnn', True),
      ('rnn_rnn_notpu', 'rnn', 'rnn', False),
      ('rnn_trf_tpu', 'rnn', 'transformer', True),
      ('rnn_trf_notpu', 'rnn', 'transformer', False),
      ('trf_rnn_tpu', 'transformer', 'rnn', True),
      ('trf_rnn_notpu', 'transformer', 'rnn', False),
      ('trf_trf_tpu', 'transformer', 'transformer', True),
      ('trf_trf_notpu', 'transformer', 'transformer', False),
  )
  def test_pors_model_encoder_decoder_type(self, encoder_type, decoder_type,
                                           use_tpu):
    FLAGS.use_tpu = use_tpu
    self.params.update({
        'add_critic': False,
        'embedding_size': 4,
        'latent_size': 4,
        'trf_hidden_size': 4,
        'trf_num_heads': 2,
        'encoder_type': encoder_type,
        'decoder_type': decoder_type,
    })
    tf.reset_default_graph()
    # Batch of 2 paragraphs each with 3 sentences.
    sids = tf.constant([
        [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
        [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]]], dtype=tf.int64)

    pids = tf.constant([[3, 2, 4, 2, 4, 2, 1, 0],
                        [5, 2, 3, 4, 2, 5, 5, 1]], dtype=tf.int64)
    features = {'sentences': sids, 'paragraphs': pids}
    ret_tensors = pors.pors_model(features, self.params, True)

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      loss, _, _, _ = ss.run(ret_tensors[:-1])
      self.assertGreater(loss, 0)

  def test_pors_model_noisy_paragraph(self):
    self.params.update({'noisy_paragraph_prob': 0.1})
    for use_tpu in [False]:
      FLAGS.use_tpu = use_tpu
      tf.reset_default_graph()
      # Batch of 2 paragraphs each with 3 sentences.
      sids = tf.constant([
          [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
          [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
      ],
                         dtype=tf.int64)

      pids = tf.constant([[3, 2, 4, 2, 4, 2, 1, 0], [5, 2, 3, 4, 2, 5, 5, 1]],
                         dtype=tf.int64)
      features = {'sentences': sids, 'paragraphs': pids,
                  'noisy_paragraphs': pids}
      ret_tensors = pors.pors_model(features, self.params, True)

      with self.session() as ss:
        ss.run(tf.initializers.global_variables())
        ss.run(tf.initializers.local_variables())
        # 2 steps to train both discriminator/generator
        loss, _, _, _ = ss.run(ret_tensors[:-1])
        self.assertGreater(loss, 0)
        loss, _, _, _ = ss.run(ret_tensors[:-1])
        self.assertGreater(loss, 0)

  @parameterized.named_parameters(
      ('use_tpu', True),
      ('no_tpu', False),
  )
  def test_pors_model_mask(self, use_tpu):
    FLAGS.use_tpu = use_tpu
    self.params.update({
        'rnn_hidden_size': 6,
        'latent_size': 4,
        'embedding_size': 10,
        'trf_hidden_size': 10,
        'tie_embeddings': True,
        'mask_rate_input': 0.1,
        'mask_prob_input': 0.1,
        'vocab_size': 11,
        'encoder_type': 'transformer',
    })
    # Batch of 2 paragraphs each with 3 sentences.
    sids = tf.constant(
        [
            [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
            [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
        ], dtype=tf.int64)

    pids = tf.constant(
        [[3, 2, 4, 2, 4, 2, 1, 0],
         [5, 2, 3, 4, 2, 5, 5, 1]], dtype=tf.int64)
    features = {'sentences': sids,
                'paragraphs': pids}
    ret_tensors = pors.pors_model(features, self.params, True,
                                  spid_dict={pors.MASK: 10})

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      loss, _, _, _ = ss.run(ret_tensors[:-1])
    self.assertGreater(loss, 0)

  def test_pors_model_tie_embeddings(self):
    tf.reset_default_graph()
    self.params.update({
        'rnn_hidden_size': 6,
        'latent_size': 4,
        'embedding_size': 3,
        'trf_hidden_size': 3,
        'tie_embeddings': True,
    })
    # Batch of 2 paragraphs each with 3 sentences.
    sids = tf.constant(
        [
            [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
            [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
        ], dtype=tf.int64)

    pids = tf.constant(
        [[3, 2, 4, 2, 4, 2, 1, 0],
         [5, 2, 3, 4, 2, 5, 5, 1]], dtype=tf.int64)
    features = {'sentences': sids,
                'paragraphs': pids}
    ret_tensors = pors.pors_model(features, self.params, True)

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      loss, _, _, _ = ss.run(ret_tensors[:-1])
    self.assertGreater(loss, 0)

  @parameterized.named_parameters(
      ('tie_emb', True),
      ('notie_emb', False),
  )
  def test_pors_different_latent(self, tie_emb):
    self.params.update({
        'rnn_hidden_size': 5,
        'latent_size': 4,
        'embedding_size': 3,
        'trf_hidden_size': 3,
        'tie_embeddings': tie_emb,
        'tie_sent_para_dec': False,
    })
    # Batch of 2 paragraphs each with 3 sentences.
    sids = tf.constant([
        [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
        [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
    ],
                       dtype=tf.int64)

    pids = tf.constant([[3, 2, 4, 2, 4, 2, 1, 0], [5, 2, 3, 4, 2, 5, 5, 1]],
                       dtype=tf.int64)
    features = {'sentences': sids, 'paragraphs': pids}
    ret_tensors = pors.pors_model(features, self.params, True)

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      loss, _, _, _ = ss.run(ret_tensors[:-1])
    self.assertGreater(loss, 0)

  def test_pors_model_tie_encs_decs(self):
    self.params.update({
        'tie_sent_para_enc': True,
        'tie_sent_para_dec': True,
        'vocab_size': 12
    })
    # Batch of 2 paragraphs each with 3 sentences.
    sids = tf.constant(
        [
            [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
            [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
        ], dtype=tf.int64)

    pids = tf.constant(
        [[3, 2, 4, 2, 4, 2, 1, 0],
         [5, 2, 3, 4, 2, 5, 5, 1]], dtype=tf.int64)
    features = {'sentences': sids,
                'paragraphs': pids}

    spid_dict = {pors.BOS: 10, pors.BOP: 11}
    ret_tensors = pors.pors_model(features, self.params, True, spid_dict)

    with self.session() as ss:
      ss.run(tf.initializers.global_variables())
      ss.run(tf.initializers.local_variables())
      loss, _, _, _ = ss.run(ret_tensors[:-1])
    self.assertGreater(loss, 0)

  @parameterized.named_parameters(
      ('use_tpu', True),
      ('no_tpu', False),
  )
  def test_input_fn(self, use_tpu):
    files = util.file_list(self.data_dir, 'valid')
    FLAGS.use_tpu = use_tpu
    input_fn = pors.get_input_fn(self.params, files, False, shuffle=False)

    dataset = input_fn({'batch_size': 2})
    it = dataset.make_one_shot_iterator()
    next_batch = it.get_next()
    with self.session() as ss:
      batch = ss.run(next_batch)
      self.assertEqual(2, batch[0]['sentences'].shape[0])

  @parameterized.named_parameters(
      ('use_tpu', True),
      ('no_tpu', False),
  )
  def test_input_fn_augmented(self, use_tpu):
    files = util.file_list(self.data_dir, 'valid')
    FLAGS.use_tpu = use_tpu
    input_fn = pors.get_input_fn(self.params, files, False, shuffle=False,
                                 roc_data_augment=True)

    dataset = input_fn({'batch_size': 2})
    it = dataset.make_one_shot_iterator()
    next_batch = it.get_next()
    with self.session() as ss:
      batch = ss.run(next_batch)
      self.assertEqual(2, batch[0]['sentences'].shape[0])
      self.assertEqual(2, batch[0]['noisy_paragraphs'].shape[0])

  @parameterized.named_parameters(
      # Rest of the tests have no critic.
      ('no_tpu_rnn', False, False, 'rnn', '', 'all'),
      ('use_tpu_rnn', True, False, 'rnn', '', 'all'),
      ('no_tpu_trf', False, False, 'transformer', '', 'all'),
      ('use_tpu_trf', True, False, 'transformer', '', 'all'),
      ('no_tpu_nsp', False, False, 'transformer', 'nsp', 'all'),
      ('use_tpu_nsp', True, False, 'transformer', 'nsp', 'all'),
      ('no_tpu_cpp', False, False, 'transformer', 'cpp', 'all'),
      ('no_tpu_train_decoder_only', False, False, 'transformer', '', 'decoder'),
  )
  def test_model_fn_smoke(self, use_tpu, add_critic, encoder_type,
                          encoder_pretraining, train_phase_subset):
    # This test a train step, running both input_fn and model_fn code paths,
    # including backward pass.
    if encoder_pretraining:
      FLAGS.in_domain_pretrain_steps = 10
    else:
      FLAGS.in_domain_pretrain_steps = 0

    FLAGS.pretrain_as_autoencoder = encoder_pretraining == 'autoencoder'

    if encoder_pretraining == 'nsp':
      self.params.update({
          'nsp_pretrain': True,
          'lambda_nsp_pretrain': 1.0,
      })
    elif encoder_pretraining == 'cpp':
      self.params.update({
          'cpp_pretrain_scheme': 'last_two',
          'lambda_cpp_pretrain': 1.0,
      })

    self.params.update({'add_critic': add_critic,
                        'train_phase_subset': train_phase_subset})
    FLAGS.use_tpu = use_tpu
    tf.reset_default_graph()
    # Just test it doesn't crash
    sptokens = [pors.BOS, pors.BOP, pors.MASK]
    tk, spid_dict = util.get_tokenizer_with_special(
        os.path.join(self.data_dir, 'wikitext103_32768.subword_vocab'),
        sptokens)
    self.params.update({
        'vocab_size': tk.vocab_size,
        'embedding_size': 4,
        'trf_hidden_size': 4,
        'trf_num_heads': 2,
        'max_decode_steps': 2,
        'encoder_type': encoder_type,
    })
    run_config = tf_estimator.tpu.RunConfig(
        model_dir=self.create_tempdir().full_path, keep_checkpoint_max=10)

    pors_estimator = tf_estimator.tpu.TPUEstimator(
        use_tpu=use_tpu,
        config=run_config,
        model_fn=pors.get_model_fn(spid_dict),
        train_batch_size=4,
        eval_batch_size=4,
        predict_batch_size=4,
        params=self.params)

    files = util.file_list(self.data_dir, 'valid')
    pors_estimator.train(input_fn=pors.get_input_fn(self.params, files, True),
                         max_steps=2)

  @parameterized.named_parameters(
      # No pretraining or critic.
      ('base', False),
      # Language model pretraining
      ('pretrain_lm_dec', True),
  )
  def test_pors_cpu_tpu_diff(self, lm_pretrain):
    self.params.update({
        'add_critic': False,
        'lm_pretrain_dec': lm_pretrain,
        'lambda_lm_pretrain_p': 1.0,
        'lambda_lm_pretrain_s': 1.0,
        'vocab_size': 13,
        'encoder_type': 'transformer',
    })
    if lm_pretrain:
      FLAGS.in_domain_pretrain_steps = 10
      FLAGS.pretrain_as_autoencoder = False
    else:
      FLAGS.in_domain_pretrain_steps = 0

    # This tests that the loss computed by cpu and tpu is the same.
    # Batch of 4 paragraphs each with 3 sentences.
    losses = {}
    for tpu in [True, False]:
      tf.reset_default_graph()
      FLAGS.use_tpu = tpu
      tf.random.set_random_seed(1234)
      sids = tf.constant([
          [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
          [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]],
          [[3, 2, 1, 0, 0], [4, 2, 4, 1, 0], [2, 1, 0, 0, 0]],
          [[5, 2, 3, 1, 0], [4, 2, 5, 1, 0], [5, 1, 0, 0, 0]]], dtype=tf.int64)
      pids = tf.constant([
          [3, 2, 4, 2, 4, 2, 1, 0], [5, 2, 3, 4, 2, 5, 5, 1],
          [3, 2, 4, 2, 4, 2, 1, 0], [5, 2, 3, 4, 2, 5, 5, 1]], dtype=tf.int64)
      features = {'sentences': sids, 'paragraphs': pids}

      ret_tensors = pors.pors_model(
          features, self.params, True,
          spid_dict={pors.MASK: 10, pors.BOS: 11, pors.BOP: 12})
      with self.session() as ss:
        ss.run(tf.initializers.global_variables())
        ss.run(tf.initializers.local_variables())
        # 2 steps to train both discriminator/generator
        loss, _, _, _ = ss.run(ret_tensors[:-1])
        losses[tpu] = loss
    self.assertAllClose(losses[True], losses[False])


if __name__ == '__main__':
  tf.test.main()
