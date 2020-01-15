# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

r"""Train a model to generate summarizing sentences for paragraphs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time


from absl import app
from absl import flags
import six
from six.moves import range
import tensorflow as tf  # tf

from summae import model as m
from summae import p2s_eval
from summae import util
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import tpu_config


FLAGS = flags.FLAGS

flags.DEFINE_enum('task', 'rocstories', ['wiki103', 'rocstories'],
                  'Which task?')
flags.DEFINE_string('data_dir', '/tmp/', 'Where tfrecord data is.')
flags.DEFINE_string('model_dir', '/tmp/test', 'Where to write model.')
flags.DEFINE_string('vocab_file', '', 'Subword vocab file.')
flags.DEFINE_integer('train_steps', 100, 'Steps to train after pretraining.'
                     'pretraining steps.')
flags.DEFINE_enum('mode', None, ['train', 'eval', 'decode', 'eval_and_decode'],
                  'Mode; for eval/decode eval_subset specifies which '
                  'data to use.')
flags.DEFINE_string('eval_subset', None,
                    'CSV of which subset (train/valid/test) to eval/decode.')
flags.DEFINE_integer('max_eval_steps', None,
                     'Max eval steps. If none, use all.')
flags.DEFINE_integer('seed', 1234, 'TF graph seed')
flags.DEFINE_integer('max_decodes', 16000, 'Maximum number of decodes.')
flags.DEFINE_bool('use_tpu', False, 'Use tpu')
flags.DEFINE_string('master', None, 'Tensorflow master URL, e.g. TPU address')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('max_sent_length', 60,
                     'Max tokens in each sentence of paragraph.'
                     'Others dropped.')
flags.DEFINE_integer('max_paragraph_length', 300,
                     'Max tokens in each paragraph. Others dropped.')
flags.DEFINE_integer('max_sent_per_paragraph', 5,
                     'Max tokens in each paragraph. Others dropped.')
flags.DEFINE_bool('decode_reconstructions', False,
                  'Whether to also output autoencoder reconstructions.')
flags.DEFINE_integer('checkpoint_steps', 5000, 'Frequency of checkpoints.')
flags.DEFINE_string('ground_truth_dir', '',
                    'Directory with {mturk,indices}.valid.csv')
flags.DEFINE_bool('debug', False, 'Debug mode.')
flags.DEFINE_bool('d_hidden_eq_hidden', False, 'Make d_hidden = hidden_size.')
flags.DEFINE_bool('pretrain_as_autoencoder', True,
                  'Autoencoder pre-training of e_p, e_s, d_p, and d_s.'
                  'If False, other pre-training methods will be applied.')
flags.DEFINE_integer('out_domain_pretrain_steps', 0,
                     'Number of steps for pretraining on out-of-domain corpus.')
flags.DEFINE_string('out_domain_data_task', 'wiki103',
                    'Which out-of-domain pretraining task?')
flags.DEFINE_string('out_domain_data_dir', '',
                    'Directory to the out-of-domain corpus.')
flags.DEFINE_integer('in_domain_pretrain_steps', 0,
                     'Number of steps for pretraining on in-domain corpus.')
flags.DEFINE_string('eval_checkpoints', None,
                    ('Csv of checkpoint numbers to eval/decode.'
                     'If not specified, continuously evals latest.'))
flags.DEFINE_string('decode_output_subdir', '', 'Output subdir')
flags.DEFINE_string('params_file', None, 'If specified, load params from path.')


# pylint: disable=invalid-name
# pylint: disable=g-long-ternary
# pylint: disable=g-long-lambda
# pylint: disable=g-complex-comprehension

# When prepending with special paragraph or sentence token, we add
# these two extra reserved tokens.
BOS = '<BOS>'
BOP = '<BOP>'
MASK = '<MASK>'


def pors_model(features, params, is_training, spid_dict=None):
  """Paragraph2Sentence, a model for unsupervised abstractive summarization.

  We use an encoder to encode either sentences or paragraphs.
  We use a decoder to decode either a sentence or paragraph, given
  a special token.
  We add a discriminator to ensure encodings of sentences/paragraphs are
  indistinguishable.

  Args:
    features: Dict of feature tensors, containing 'sentences' and 'paragraphs'
    params: hyper-parameters dict, including 'batch_size'
    is_training: whether in train mode
    spid_dict: dictionary of extra_token_strings to vocab ids

  Returns:
    loss_r: scalar tensor loss
    train_op: model train op
    metric_ops_dict: dict of eval metrics, for evaluate mode
    preds_dict: dict of predictions, for predict mode
  """
  # Warning: do not use any FLAGS here so that all model configuration can be
  # loaded # from hypers.json.
  s_ids_BxNxL = features['sentences']
  p_ids_BxS = features['paragraphs']
  batch_size = p_ids_BxS.get_shape()[0]
  assert isinstance(params, dict)
  tf.logging.info(params)
  embedding_size = params['embedding_size']
  learning_rate = params['learning_rate']
  tie_embeddings = params['tie_embeddings']
  tie_encs = params['tie_sent_para_enc']
  tie_decs = params['tie_sent_para_dec']
  latent_size = params['latent_size']
  vocab_size = params['vocab_size']
  add_critic = params['add_critic']
  max_paragraph_length = params['max_paragraph_length']
  max_sent_length = params['max_sent_length']

  # Masking hypers.
  # The masking proceeds as follows (only during training):
  #   1. Decide whether to mask input with probability max_prob_input.
  #   2. If masking, mask each token independently with probability
  #   mask_rate_input.
  mask_prob_input = params['mask_prob_input']
  mask_rate_input = params['mask_rate_input']
  mask_sentences = ((params['mask_type'] == 'sentences' or
                     params['mask_type'] == 'both') and
                    mask_rate_input > 0 and mask_prob_input > 0)
  mask_paragraphs = ((params['mask_type'] == 'paragraphs' or
                      params['mask_type'] == 'both') and
                     mask_rate_input > 0 and mask_prob_input > 0)

  # Pre-training hypers.
  total_pretrain_steps = (params['out_domain_pretrain_steps'] +
                          params['in_domain_pretrain_steps'])
  # When total_pretrain_steps == 0, the pre-training configs won't affect the
  # training.
  nsp_pretrain = params['nsp_pretrain']
  cpp_pretrain_scheme = params['cpp_pretrain_scheme']
  lm_pretrain_dec = params['lm_pretrain_dec']
  # The following checks rule out some undesired pre-training scenarios.
  if total_pretrain_steps > 0:
    none_ae_pretrain_is_on = (nsp_pretrain or
                              cpp_pretrain_scheme or lm_pretrain_dec)
    if params['pretrain_as_autoencoder']:
      assert not none_ae_pretrain_is_on, ('Pre-training method conflict! When '
                                          'pre-training as autoencoder, make '
                                          'sure none of the other strategies is'
                                          ' on.')
    else:  # non-autoencoder pre-training
      assert none_ae_pretrain_is_on, ('No pre-training action! When we want to '
                                      'pre-train with strategies other than '
                                      'autoencoding, make sure at least one of '
                                      'them is on.')

  if tie_encs or tie_decs:
    assert spid_dict
    bos_id = spid_dict[BOS]
    bop_id = spid_dict[BOP]

  # When Z != H,
  # 1. We add a projection matrix to transform encoder hidden size, H, to Z.
  # 2. The conditioning vector is concatenated at each time step of the decoder.
  embedding_VxE = tf.get_variable(
      name='ae_token_embedding', shape=[vocab_size, embedding_size],
      initializer=tf.truncated_normal_initializer(stddev=m.SMALL_WEIGHTS_SD))

  if (mask_sentences or mask_paragraphs):
    assert spid_dict
    mask_id = spid_dict[MASK]
    assert mask_id
    mask_emb_E = tf.nn.embedding_lookup(embedding_VxE, mask_id)

  # Four primary architecture components:
  #   1. sentence encoder, e_s
  #   2. sentence decoder, d_s
  #   3. paragraph encoder, d_s
  #   4. paragraph decoder, d_p

  encoder_type = params['encoder_type']
  assert encoder_type in ('rnn', 'transformer'), encoder_type
  tf.logging.info('Using %s as encoder.', encoder_type)
  if encoder_type == 'rnn':
    cond_only_init = latent_size == params['rnn_hidden_size']  # i.e., Z == H
    e_p = m.GruEncoder(
        hidden_size=params['rnn_hidden_size'],
        num_layers=params['rnn_num_layers'],
        latent_size=latent_size,
        scope='ae_paragraph_encoder',
        bidirect_encode=params['rnn_bidirect_encode'])
    e_s = e_p if tie_encs else m.GruEncoder(
        hidden_size=params['rnn_hidden_size'],
        num_layers=params['rnn_num_layers'],
        latent_size=latent_size,
        scope='ae_sentence_encoder',
        bidirect_encode=params['rnn_bidirect_encode'])
    pooling = params['rnn_pooling']
  else:
    # Transformer-based encoder has more constraints on hypers.
    assert params['trf_hidden_size'] == embedding_size
    assert params['trf_hidden_size'] % params['trf_num_heads'] == 0
    assert params['trf_hidden_size'] % 2 == 0
    cond_only_init = latent_size == params['trf_hidden_size']  # i.e., Z == H
    e_p = m.TransformerEncoder(
        num_layers=params['trf_num_layers'],
        num_heads=params['trf_num_heads'],
        hidden_size=params['trf_hidden_size'],
        filter_size=params['trf_filter_size'],
        attention_dropout=params['trf_attention_dropout'],
        relu_dropout=params['trf_relu_dropout'],
        postprocess_dropout=params['trf_postprocess_dropout'],
        latent_size=latent_size,
        scope='ae_paragraph_encoder')
    e_s = e_p if tie_encs else m.TransformerEncoder(
        num_layers=params['trf_num_layers'],
        num_heads=params['trf_num_heads'],
        hidden_size=params['trf_hidden_size'],
        filter_size=params['trf_filter_size'],
        attention_dropout=params['trf_attention_dropout'],
        relu_dropout=params['trf_relu_dropout'],
        postprocess_dropout=params['trf_postprocess_dropout'],
        latent_size=latent_size,
        scope='ae_sentence_encoder')
    pooling = params['trf_pooling']

  decoder_type = params['decoder_type']
  assert decoder_type in ('rnn', 'transformer'), decoder_type
  tf.logging.info('Using %s as decoder.', decoder_type)
  if decoder_type == 'rnn':
    d_p = m.GruDecoder(
        hidden_size=params['rnn_hidden_size'],
        vocab_size=vocab_size,
        embed_VxE=embedding_VxE,
        max_steps=max_paragraph_length,
        num_layers=params['rnn_num_layers'],
        tie_embeddings=tie_embeddings,
        scope='ae_paragraph_decoder',
        cond_only_init=cond_only_init)
    d_s = d_p if tie_decs else m.GruDecoder(
        params['rnn_hidden_size'],
        vocab_size=vocab_size,
        embed_VxE=embedding_VxE,
        max_steps=max_sent_length + 10,
        num_layers=params['rnn_num_layers'],
        tie_embeddings=tie_embeddings,
        scope='ae_sentence_decoder',
        cond_only_init=cond_only_init)
  else:
    # Transformer-based decoder.
    assert params['trf_hidden_size'] == embedding_size
    assert params['trf_hidden_size'] % params['trf_num_heads'] == 0
    assert params['trf_hidden_size'] % 2 == 0
    cond_by_addition = latent_size == embedding_size
    d_p = m.TransformerDecoder(
        num_layers=params['trf_num_layers'],
        num_heads=params['trf_num_heads'],
        hidden_size=params['trf_hidden_size'],
        filter_size=params['trf_filter_size'],
        attention_dropout=params['trf_attention_dropout'],
        relu_dropout=params['trf_relu_dropout'],
        postprocess_dropout=params['trf_postprocess_dropout'],
        embed_VxE=embedding_VxE,
        vocab_size=vocab_size,
        max_steps=max_paragraph_length,
        latent_size=latent_size,
        tie_embeddings=tie_embeddings,
        cond_by_addition=cond_by_addition,
        scope='ae_paragraph_decoder')
    d_s = d_p if tie_decs else m.TransformerDecoder(
        num_layers=params['trf_num_layers'],
        num_heads=params['trf_num_heads'],
        hidden_size=params['trf_hidden_size'],
        filter_size=params['trf_filter_size'],
        attention_dropout=params['trf_attention_dropout'],
        relu_dropout=params['trf_relu_dropout'],
        postprocess_dropout=params['trf_postprocess_dropout'],
        embed_VxE=embedding_VxE,
        vocab_size=vocab_size,
        max_steps=max_sent_length + 10,
        latent_size=latent_size,
        tie_embeddings=tie_embeddings,
        cond_by_addition=cond_by_addition,
        scope='ae_sentence_decoder')

  # Sentence autoencoder.
  # Sentence input features: s_ids_BxNxL
  # Convert sentences into a sentence superbatch.
  # If L is fixed, say on TPU, we can instead use tf.reshape()
  s_ids_YxL = tf.concat(tf.unstack(s_ids_BxNxL, axis=0), 0)
  s_seq_lengths_Y = m.id_seq_length(s_ids_YxL)
  # Keep s_enc_inputs_YxLxE unchanged so that it can be reused below.
  s_enc_inputs_YxLxE = tf.nn.embedding_lookup(embedding_VxE, s_ids_YxL)
  # Teacher forcing uses the original inputs; prepend special tokens and remove
  # the last token if tie_decs.
  s_dec_inputs_YxLxE = (
      m.prepend_token(s_enc_inputs_YxLxE[:, :-1, :], embedding_VxE, bos_id)
      if tie_decs else m.shift_right_3d(s_enc_inputs_YxLxE))
  # TODO(peterjliu): Add special token for encoder as well.

  s_enc_training_inputs_YxLxE = s_enc_inputs_YxLxE
  # TODO(peterjliu): Randomly select a subset of sentences to mask.
  if is_training and mask_sentences:
    tf.logging.info('Add sentence masking to training phase.')
    # Randomly mask tokens in the input.
    rand_f = tf.random.uniform([])
    s_enc_training_inputs_YxLxE = tf.cond(
        rand_f < mask_prob_input,
        lambda: m.mask_embs(s_enc_training_inputs_YxLxE, s_seq_lengths_Y,
                            mask_rate_input, mask_emb_E),
        lambda: tf.identity(s_enc_training_inputs_YxLxE),
        name='cond_sentence_mask')

  s_enc_YxZ, _ = e_s.encode(
      s_enc_training_inputs_YxLxE, s_seq_lengths_Y, pooling, is_training)

  # *tf for 'teacher-force'
  stf_enc_YxZ = s_enc_YxZ
  s_logits_YxLxV = d_s.teacher_force(
      stf_enc_YxZ, s_dec_inputs_YxLxE, s_seq_lengths_Y)
  r_s, s_metrics = m.ce_loss(s_logits_YxLxV, s_ids_YxL, s_seq_lengths_Y)

  # Paragraph autoencoder.
  # Paragraph input features: p_ids_BxS
  p_seq_lengths_B = m.id_seq_length(p_ids_BxS)
  # Keep p_enc_inputs_BxSxE unchanged as well.
  p_enc_inputs_BxSxE = tf.nn.embedding_lookup(embedding_VxE, p_ids_BxS)
  p_dec_inputs_BxSxE = (
      m.prepend_token(p_enc_inputs_BxSxE[:, :-1, :], embedding_VxE, bop_id)
      if tie_decs else m.shift_right_3d(p_enc_inputs_BxSxE))

  p_enc_training_inputs_BxSxE = p_enc_inputs_BxSxE
  noisy_paragraph_prob = params['noisy_paragraph_prob']
  if is_training and noisy_paragraph_prob > 0:
    tf.logging.info('Apply paragraph shuffling noise to training phase.')
    p_shuffled_ids_BxS = features['noisy_paragraphs']
    p_shuffled_BxSxE = tf.nn.embedding_lookup(embedding_VxE, p_shuffled_ids_BxS)
    # p_seq_lengths_B should be the same
    randp_f = tf.random.uniform([])
    # With probability noisy_paragraph_prob, replace original paragraph
    # with noisy version.
    p_enc_training_inputs_BxSxE = tf.cond(
        randp_f < noisy_paragraph_prob,
        lambda: p_shuffled_BxSxE,
        lambda: p_enc_inputs_BxSxE,
        name='cond_paragraph_shuffle')

  if is_training and mask_paragraphs:
    tf.logging.info('Add paragraph masking to training phase.')
    rand2_f = tf.random.uniform([])
    p_enc_training_inputs_BxSxE = tf.cond(
        rand2_f < mask_prob_input,
        lambda: m.mask_embs(p_enc_training_inputs_BxSxE, p_seq_lengths_B,
                            mask_rate_input, mask_emb_E),
        lambda: tf.identity(p_enc_training_inputs_BxSxE),
        name='cond_paragraph_mask')

  p_enc_BxZ, _ = e_p.encode(
      p_enc_training_inputs_BxSxE, p_seq_lengths_B, pooling, is_training)
  ptf_enc_BxZ = p_enc_BxZ
  p_logits_BxSxV = d_p.teacher_force(
      ptf_enc_BxZ, p_dec_inputs_BxSxE, p_seq_lengths_B)
  r_p, p_metrics = m.ce_loss(p_logits_BxSxV, p_ids_BxS, p_seq_lengths_B)

  # Reconstruction loss of sentence and paragraph autoencoders.
  loss_r = params['lambda_s'] * r_s + params['lambda_p'] * r_p

  # Assign ae_pretrain_loss here so that it only includes r_p and r_s. This
  # pre-train loss will only be used when pretrain_as_autoencoder is True.
  ae_pretrain_loss = loss_r

  # Decoder unconditional LM pre-training.
  lm_pretrain_loss_s = tf.zeros([])
  lm_pretrain_loss_p = tf.zeros([])
  if lm_pretrain_dec:
    tf.logging.info('Pre-train the decoder as an unconditional language model '
                    'by conditioning on zero vectors.')
    ptf_pretrain_BxZ = tf.zeros_like(p_enc_BxZ)
    p_ulm_logits_BxSxV = d_p.teacher_force(
        ptf_pretrain_BxZ, p_dec_inputs_BxSxE, p_seq_lengths_B)
    lm_pretrain_loss_p, lm_pretrain_p_metrics = m.ce_loss(
        p_ulm_logits_BxSxV, p_ids_BxS, p_seq_lengths_B)

    stf_pretrain_YxZ = tf.zeros_like(s_enc_YxZ)
    s_ulm_logits_YxLxV = d_s.teacher_force(
        stf_pretrain_YxZ, s_dec_inputs_YxLxE, s_seq_lengths_Y)
    lm_pretrain_loss_s, lm_pretrain_s_metrics = m.ce_loss(
        s_ulm_logits_YxLxV, s_ids_YxL, s_seq_lengths_Y)

  # Next sentence prediction (nsp) pre-training.
  nsp_pretrain_loss = tf.zeros([])
  if nsp_pretrain:
    tf.logging.info('Add next sentence prediction (NSP) as encoder pre-training'
                    ' task. Only teaches the encoder to encode sentences.')
    not_next_diff_p_prob = params['nsp_pretrain_not_next_diff_paragraph_prob']
    nsp_pretrain_loss, nsp_logits_2B, nsp_labels_2B = (
        m.compute_nsp_pretrain_loss(e_s, s_enc_inputs_YxLxE, s_seq_lengths_Y,
                                    pooling, is_training, batch_size,
                                    not_next_diff_p_prob))

  # Corrupted paragraph prediction (CPP) pre-training.
  cpp_pretrain_loss = tf.zeros([])
  if cpp_pretrain_scheme:
    tf.logging.info('Add corrupted paragraph prediction (CPP) as pre-training '
                    'task with paragraph corrupted scheme: %s. Only teaches the'
                    ' encoder to encode paragraphs.', cpp_pretrain_scheme)
    cpp_pretrain_loss, cpp_logits_B, cpp_labels_B = (
        m.compute_cpp_pretrain_loss(e_p, s_ids_BxNxL, embedding_VxE, pooling,
                                    is_training, scheme=cpp_pretrain_scheme,
                                    p_enc_inputs_BxSxE=p_enc_inputs_BxSxE,
                                    p_seq_lengths_B=p_seq_lengths_B))

  # We start collecting training loss here, including ae_loss and disc_loss.
  ae_loss = loss_r          # autoencoder loss
  disc_loss = tf.zeros([])  # discriminator loss

  # We want the avg. distances between paragraph embeddings and their sentence
  # embeddings to be small.
  if params['lambda_c_avg2'] > 0.0:
    tf.logging.info('Add regularizer to loss which is the average cosine '
                    'distance between paragraph embeddings and their sentence '
                    'embeddings.')
    loss_c_avg2 = m.compute_c_avg2_loss(s_enc_YxZ, p_enc_BxZ)
    ae_loss += params['lambda_c_avg2'] * loss_c_avg2

  if add_critic:
    # Adversarial regularizer.
    # We want to encode paragraphs/sentences such that a discriminator cannot
    # distinguish whether they are paragraphs/sentences. Sentence/Paragraph
    # encodings have label 0/1 respectively.
    tf.logging.info('Adding adversarial loss, weight %d', params['adv_weight'])
    d_loss, adv_loss, d_logits_2B, d_labels_2B = m.compute_critic_loss(
        s_enc_YxZ, p_enc_BxZ, n_hiddens=params['d_hidden'], scope='disc_')
    ae_loss += params['adv_weight'] * adv_loss
    disc_loss += d_loss

  # First token to prepend when decoding a sentence from the paragraph.
  p2s_first_token = bos_id if tie_decs else -1
  global_step = tf.train.get_or_create_global_step()
  # i.e., after pre-training
  training_phase = tf.greater(global_step, total_pretrain_steps)

  if params['pretrain_as_autoencoder']:
    pretrain_loss = ae_pretrain_loss
  else:
    encoder_pretrain_loss = (
        params['lambda_nsp_pretrain'] * nsp_pretrain_loss +
        params['lambda_cpp_pretrain'] * cpp_pretrain_loss)
    decoder_pretrain_loss = (
        params['lambda_lm_pretrain_s'] * lm_pretrain_loss_s +
        params['lambda_lm_pretrain_p'] * lm_pretrain_loss_p)

    # Sequential pre-training of encoder and decoder.
    # The component (encoder or decoder) specified as by pretrain_order will
    # first be pre-trained for first_pretrain_steps steps, and the rest of the
    # pre-training steps (total_pretrain_steps - first_pretrain_steps) will
    # pre-train the second component.
    pretrain_order = params['pretrain_order']
    first_pretrain_steps = params['first_pretrain_steps']
    # The following two assertions rule out some undesired scenarios during
    # sequential pre-training although without them the training can still run.
    if pretrain_order in ('encoder_first', 'decoder_first'):
      assert (first_pretrain_steps < total_pretrain_steps and
              first_pretrain_steps > 0), ('When doing encoder and decoder '
                                          'sequential pre-training, '
                                          'first_pretrain_steps should be '
                                          'within (0, total_pretrain_steps).')
      encoder_pretraining = (nsp_pretrain or cpp_pretrain_scheme)
      assert (lm_pretrain_dec and encoder_pretraining), ('Sequential '
                                                         'pre-training is on '
                                                         'but one of encoder '
                                                         'and decoder '
                                                         'pre-training is off. '
                                                         'This loses the '
                                                         'meaning of doing '
                                                         'sequential '
                                                         'pre-training.')
      first_do, first_pretrain_loss, second_do, second_pretrain_loss = (
          ('encoder', encoder_pretrain_loss, 'decoder', decoder_pretrain_loss)
          if pretrain_order == 'encoder_first' else
          ('decoder', decoder_pretrain_loss, 'encoder', encoder_pretrain_loss))
      tf.logging.info('Sequential pre-training is on! Pre-train the %s for %d '
                      'steps first and then %s for %d steps.',
                      first_do, first_pretrain_steps,
                      second_do, total_pretrain_steps - first_pretrain_steps)
      pretrain_loss = tf.cond(
          tf.greater(global_step, first_pretrain_steps),
          lambda: second_pretrain_loss,
          lambda: first_pretrain_loss)
    elif pretrain_order == 'simultaneous':
      # encoder and decoder will be pre-trained jointly.
      pretrain_loss = encoder_pretrain_loss + decoder_pretrain_loss
    else:
      tf.logging.fatal('pretrain_order should be one of encoder_first, '
                       'decoder_first, and simultaneous.')

  # phase_loss defines the non-discriminator loss
  # to optimize in the current training phase,
  # i.e., pre-training loss or regular training loss.
  phase_loss = tf.cond(
      training_phase,
      lambda: ae_loss,
      lambda: pretrain_loss)

  ae_opt = m.get_adam(learning_rate, FLAGS.use_tpu)
  all_ae_vars = tf.global_variables('ae_')
  if add_critic:
    tf.logging.info('Add critic(s).')
    # Note previous to cl/250371333, d-step ran even during pre-training,
    # although adversarial loss was ignored.
    # Be careful to construct train_op within cond function, and not before it.
    do_discrim_step = tf.logical_and(
        training_phase,
        tf.equal(0, tf.to_int32(tf.mod(global_step,
                                       params['gd_step_ratio'] + 1))))
    d_opt = m.get_adam(learning_rate, FLAGS.use_tpu)
    d_vars = tf.global_variables('disc_') + tf.global_variables('prior_disc_')
    train_op = tf.cond(
        do_discrim_step,
        lambda: d_opt.minimize(disc_loss, global_step, var_list=d_vars),
        lambda: ae_opt.minimize(phase_loss,
                                global_step,
                                var_list=all_ae_vars))
  else:
    # In non-critic models we support only updating part of the model in the
    # training phase.
    # Collect trainable variables for autoencoders.
    if params['train_phase_subset'] == 'all':
      train_op = ae_opt.minimize(phase_loss, global_step, var_list=all_ae_vars)
    else:
      tf.logging.info('train subset of params')
      if params['train_phase_subset'] == 'decoder':
        tf.logging.info('Only update decoder params in train phase.')
        train_ae_vars = tf.global_variables('ae_.*decoder.*')
      elif params['train_phase_subset'] == 'decoder_and_embedding':
        tf.logging.info(
            'Only update decoder and embedding params in train phase.')
        train_ae_vars = (tf.global_variables('ae_.*decoder.*') +
                         tf.global_variables('ae_token_embedding'))
      else:
        tf.logging.fatal('Invalid train_phase_subset: %s',
                         params['train_phase_subset'])
      assert train_ae_vars, 'No train vars to update.'

      # TODO(peterjliu): Get this working on TPU
      train_op = tf.cond(training_phase,
                         lambda: ae_opt.minimize(phase_loss, global_step,
                                                 var_list=train_ae_vars),
                         # During pretraining, all ae vars are adjusted.
                         lambda: ae_opt.minimize(phase_loss, global_step,
                                                 var_list=all_ae_vars))

  # Decode sentence from paragraph encoding.
  decoding_method = params['decoding_method']
  decoding_beam_size = params['decoding_beam_size']
  decoding_alpha = params['decoding_alpha']
  p2s_enc_BxZ = p_enc_BxZ
  p2s_dec_ar_BxU = d_s.decode_v(
      p2s_enc_BxZ,
      method=decoding_method,
      first_token=p2s_first_token,
      max_steps=max_sent_length + 10,
      beam_size=decoding_beam_size,
      alpha=decoding_alpha)
  p2s_length_B = m.id_seq_length(p2s_dec_ar_BxU)
  p2s_avg_seq_length = tf.reduce_mean(p2s_length_B)

  if FLAGS.use_tpu:
    def host_call_fn(gs, d_loss, s_loss, p_loss, lr, loss_rec):
      """Processes tensors exported from TPU to host."""
      with summary.create_file_writer(FLAGS.model_dir).as_default():
        with summary.always_record_summaries():
          gs = gs[0]
          summary.scalar('d_loss', d_loss[0], step=gs)
          summary.scalar('s_loss', s_loss[0], step=gs)
          summary.scalar('p_loss', p_loss[0], step=gs)
          summary.scalar('loss_reconstruction_total', loss_rec[0], step=gs)
          summary.scalar('learning_rate', lr[0], step=gs)
          return summary.all_summary_ops()
    host_t = [tf.reshape(t, [1]) for t in [
        global_step, disc_loss, r_s, r_p, learning_rate, loss_r]]
    host_call = (host_call_fn, host_t)
  else:
    host_call = None
    # TODO(peterjliu): Figure out how to do these on TPU
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('s_loss', r_s)
    tf.summary.scalar('s_acc', s_metrics['accuracy'][1])
    tf.summary.scalar('p_loss', r_p)
    tf.summary.scalar('p_acc', p_metrics['accuracy'][1])
    tf.summary.scalar('p_avg_seq_length', tf.reduce_mean(p_seq_lengths_B))
    tf.summary.scalar('s_avg_seq_length', tf.reduce_mean(s_seq_lengths_Y))
    tf.summary.scalar('p2s_avg_seq_length', p2s_avg_seq_length)

    if total_pretrain_steps > 0:  # Only show pre-training metrics when it's on.
      tf.summary.scalar('pretrain_loss', pretrain_loss)

      if nsp_pretrain:
        nsp_predict_2B = tf.math.greater(tf.math.sigmoid(nsp_logits_2B), 0.5)
        nsp_acc = tf.metrics.accuracy(labels=nsp_labels_2B,
                                      predictions=nsp_predict_2B)
        tf.summary.scalar('nsp_pretrain_acc', nsp_acc[1])
        tf.summary.scalar('nsp_pretrain_loss', nsp_pretrain_loss)

      if cpp_pretrain_scheme:
        cpp_predict_B = tf.math.greater(tf.math.sigmoid(cpp_logits_B), 0.5)
        cpp_acc = tf.metrics.accuracy(labels=cpp_labels_B,
                                      predictions=cpp_predict_B)
        tf.summary.scalar('cpp_pretrain_acc', cpp_acc[1])
        tf.summary.scalar('cpp_pretrain_loss', cpp_pretrain_loss)

      if lm_pretrain_dec:
        tf.summary.scalar('lm_pretrain_loss_s', lm_pretrain_loss_s)
        tf.summary.scalar('lm_pretrain_acc_s',
                          lm_pretrain_s_metrics['accuracy'][1])
        tf.summary.scalar('lm_pretrain_loss_p', lm_pretrain_loss_p)
        tf.summary.scalar('lm_pretrain_acc_p',
                          lm_pretrain_p_metrics['accuracy'][1])

    if params['lambda_c_avg2'] > 0.0:
      tf.summary.scalar('loss_c_avg2', loss_c_avg2)

    if add_critic:
      d_predict_2B = tf.math.greater(tf.math.sigmoid(d_logits_2B), 0.5)
      d_acc = tf.metrics.accuracy(labels=d_labels_2B, predictions=d_predict_2B)
      tf.summary.scalar('d_acc', d_acc[1])
      tf.summary.scalar('d_loss', d_loss)
      tf.summary.scalar('adv_loss', adv_loss)

  metric_ops_dict = {}
  preds_dict = {}
  if not FLAGS.use_tpu:  # Currently we only run eval on GPU
    # Eval metrics
    metric_ops_dict = {
        'p_accuracy_next_token': p_metrics['accuracy'],
        's_accuracy_next_token': s_metrics['accuracy'],
        'p_loss': tf.metrics.mean(r_p),
        's_loss': tf.metrics.mean(r_s),
        'p2s_avg_seq_length': tf.metrics.mean(p2s_length_B)
    }

    if total_pretrain_steps > 0:
      metric_ops_dict.update({
          'pretrain_loss': tf.metrics.mean(pretrain_loss)
      })

      if nsp_pretrain:
        metric_ops_dict.update({
            'nsp_pretrain_loss': tf.metrics.mean(nsp_pretrain_loss),
            'nsp_pretrain_acc': nsp_acc
        })

      if cpp_pretrain_scheme:
        metric_ops_dict.update({
            'cpp_pretrain_loss': tf.metrics.mean(cpp_pretrain_loss),
            'cpp_pretrain_acc': cpp_acc
        })

      if lm_pretrain_dec:
        metric_ops_dict.update({
            'lm_pretrain_s_acc_next_token': lm_pretrain_s_metrics['accuracy'],
            'lm_pretrain_p_acc_next_token': lm_pretrain_p_metrics['accuracy'],
            'lm_pretrain_loss_s': tf.metrics.mean(lm_pretrain_loss_s),
            'lm_pretrain_loss_p': tf.metrics.mean(lm_pretrain_loss_p)
        })

    if params['lambda_c_avg2'] > 0.0:
      metric_ops_dict.update({
          'loss_c_avg2': tf.metrics.mean(loss_c_avg2)
      })

    if add_critic:
      metric_ops_dict.update({
          'd_loss': tf.metrics.mean(d_loss),
          'd_accuracy': d_acc,
          'adv_loss': tf.metrics.mean(adv_loss)
      })

    # Dict of tensors to pass to EstimatorSpec in PREDICT mode.
    preds_dict = {
        'summary': p2s_dec_ar_BxU, 'document': p_ids_BxS,
        'first_sentence': s_ids_BxNxL[:, 0, :],
        'paragraph_encodings': p_enc_BxZ,
        'sentence_encodings': tf.reshape(s_enc_YxZ,
                                         [batch_size, -1, latent_size]),
        'paragraph_lengths': p_seq_lengths_B,
        'sentence_lengths': tf.reshape(s_seq_lengths_Y, [batch_size, -1])}
    if FLAGS.decode_reconstructions:
      tf.logging.info('will decode reconstructions')
      # Only for predict mode, decode auto-regressively to see how good
      # reconstructions are:
      # predict() requires tensors to have all same batch size.

      s_first_token = bos_id if tie_decs else -1
      p_first_token = bop_id if tie_decs else -1

      # Sentence2sentence reconstruction.
      s_enc_BxNxZ = tf.reshape(s_enc_YxZ, [batch_size, -1, latent_size])
      # Just decode first sentence of paragraph.
      s_enc_BxZ = tf.squeeze(s_enc_BxNxZ[:, 0, :])
      sv_enc_BxZ = s_enc_BxZ
      pv_enc_BxZ = p_enc_BxZ
      s_dec_ar_BxU = d_s.decode_v(sv_enc_BxZ, method=decoding_method,
                                  first_token=s_first_token,
                                  beam_size=decoding_beam_size,
                                  alpha=decoding_alpha)
      # Paragraph2paragraph reconstructions.
      p_dec_ar_BxU = d_p.decode_v(pv_enc_BxZ, method=decoding_method,
                                  first_token=p_first_token,
                                  beam_size=decoding_beam_size,
                                  alpha=decoding_alpha)
      # Sentence2paragraph expansions from the first sentence.
      s2p_dec_ar_BxU = d_p.decode_v(sv_enc_BxZ, method=decoding_method,
                                    first_token=p_first_token,
                                    beam_size=decoding_beam_size,
                                    alpha=decoding_alpha)
      preds_dict.update({
          'decoded_paragraph': p_dec_ar_BxU,
          'decoded_sentences': s_dec_ar_BxU,
          'expanded_paragraphs': s2p_dec_ar_BxU,
      })
  return loss_r, train_op, metric_ops_dict, preds_dict, host_call


# Actual model_fn for estimator
# pylint: disable=unused-argument
def get_model_fn(spid_dict):
  """Returns model_fn."""
  def model_fn(features, labels, mode, params):
    """Autoencoder model function.

    Args:
      features: a dict of tensors, with 'sentences', 'paragraphs'
      labels: needed by estimator API, ignored here
      mode: tf estimator mode
      params: dict of hyperparams
    """
    loss, train_op, metrics_dict, preds_dict, host_call = pors_model(
        features, params, mode == tf.estimator.ModeKeys.TRAIN, spid_dict)

    # Need loss, train_op
    if mode == tf.estimator.ModeKeys.TRAIN:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op, host_call=host_call)
    elif mode == tf.estimator.ModeKeys.EVAL:
      # We need to use eval_on_tpu=False when creating TPUEstimator for
      # this to work.
      # TODO(peterjliu): Get eval on TPU?
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                        eval_metric_ops=metrics_dict)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      # Use this mode for auto-regressive decoding.
      return tf.contrib.tpu.TPUEstimatorSpec(mode,
                                             predictions=preds_dict)
  return model_fn


def get_input_fn(params, data_files, repeat=False, shuffle=True,
                 roc_data_augment=False):
  """Returns input_fn.

  For the max_* parameters, examples are dropped if exceeding them.

  Args:
    params: dictionary of experiment parameters
    data_files: list of filenames
    repeat: whether to repeat indefinitely (for training)
    shuffle: whether to shuffle data
    roc_data_augment: whether to add 'noisy_paragraphs' feature

  Returns:
    input_fn for Estimator
  """
  max_sent_per_paragraph = params['max_sent_per_paragraph']
  max_paragraph_length = params['max_paragraph_length']
  max_sent_length = params['max_sent_length']
  augment_scheme = params['noisy_paragraph_scheme']
  def small_enough(sp_x):
    # If true, keep the example.
    return tf.logical_and(
        tf.logical_and(
            # Too many sentences.
            tf.less_equal(tf.shape(sp_x)[0], max_sent_per_paragraph),
            # A sentence is too long.
            tf.less_equal(tf.shape(sp_x)[1], max_sent_length)),
        # Paragraph too long.
        tf.less_equal(tf.size(sp_x.values), max_paragraph_length))

  def parse_seqex(seqex_pb):
    features = {'sentences': tf.VarLenFeature(dtype=tf.int64)}
    _, parsed_features = tf.parse_single_sequence_example(
        seqex_pb, sequence_features=features)
    return parsed_features['sentences']

  def add_noise(x_d, augment_scheme):
    if augment_scheme == 'shuffle_sentences':
      return tf.random.shuffle(x_d)
    else:
      assert FLAGS.task == 'rocstories'
      # swap_neighbors
      return m.swap_neighboring_rows_5(x_d)

  def input_fn(params):
    """Sentence and paragraph input_fn for p2s model.

    Args:
      params: dictionary of parameters

    Returns:
      dataset of (feature_dict, None), former has 'sentences' and 'paragraphs'
    """
    batch_size = params['batch_size']
    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.map(parse_seqex)  # sparse ids of paragraph
    dataset = dataset.filter(small_enough)
    # TODO(peterjliu): Shuffle better
    if shuffle:
      dataset = dataset.shuffle(buffer_size=10000,
                                reshuffle_each_iteration=True)
    if repeat:
      dataset = dataset.repeat()

    dataset = dataset.map(lambda x: (x, tf.sparse.to_dense(x)))
    if roc_data_augment:
      dataset = dataset.map(
          lambda x, x_d: (
              x,
              x_d,
              # Like x, but shuffled along first dim
              # TODO(peterjliu): Different shuffling schemes.
              tf.contrib.layers.dense_to_sparse(add_noise(x_d, augment_scheme))
          )
      )
      dataset = dataset.map(lambda x, x_d, x_r: (
          # 'features'
          {
              # paragraph in 2-d, with one sentence per row
              'sentences': m.add_eos_2d(x_d),
              # 1-d flatttened paragraph tensor
              'paragraphs': tf.concat([x.values, [util.EOS_ID]], axis=0),
              'noisy_paragraphs': tf.concat([x_r.values, [util.EOS_ID]],
                                            axis=0),
          },
          # 'labels', unused but Estimator requires something
          tf.constant(0)))
      if FLAGS.use_tpu:
        # TPU requires fixed shapes, so pad to size
        return dataset.padded_batch(
            batch_size,
            padded_shapes=(
                {
                    'sentences': [max_sent_per_paragraph,
                                  max_sent_length + 1],  # + EOS
                    'paragraphs': [max_paragraph_length + 1],
                    'noisy_paragraphs': [max_paragraph_length + 1]
                },
                []),
            drop_remainder=True)
      else:
        return dataset.padded_batch(
            batch_size,
            padded_shapes=({'sentences': [None, None],
                            'paragraphs': [None],
                            'noisy_paragraphs': [None]
                           },
                           []),
            drop_remainder=True)
    else:
      dataset = dataset.map(lambda x, x_d: (
          # 'features'
          {
              # paragraph in 2-d, with one sentence per row
              'sentences': m.add_eos_2d(x_d),
              # 1-d flatttened paragraph tensor
              'paragraphs': tf.concat([x.values, [util.EOS_ID]], axis=0)
          },
          # 'labels', unused but Estimator requires something
          tf.constant(0)))

      if FLAGS.use_tpu:
        # TPU requires fixed shapes, so pad to size
        return dataset.padded_batch(
            batch_size,
            padded_shapes=(
                {
                    'sentences': [max_sent_per_paragraph,
                                  max_sent_length + 1],  # +EOs
                    'paragraphs': [max_paragraph_length + 1]
                },
                []),
            drop_remainder=True)
      else:
        return dataset.padded_batch(
            batch_size,
            padded_shapes=({'sentences': [None, None],
                            'paragraphs': [None]},
                           []),
            drop_remainder=True)
  return input_fn


def write_line(f, out):
  # TODO(peterjliu): Consider deleting this.
  tf.logging.info(out)
  f.write(out + '\n')


def first_sentence(arr):
  s = arr.tolist()
  for i in range(len(s)):
    if s[i] == 1:
      end_ind = i
  return tuple(s[:end_ind])


def flags_hypers():
  # Dictionary of params that are specified through top-level flags.
  return {
      # Note batch_size is reserved and should not be included.
      'd_hidden_eq_hidden': FLAGS.d_hidden_eq_hidden,
      'in_domain_pretrain_steps': FLAGS.in_domain_pretrain_steps,
      'out_domain_pretrain_steps': FLAGS.out_domain_pretrain_steps,
      'pretrain_as_autoencoder': FLAGS.pretrain_as_autoencoder,
      'max_paragraph_length': FLAGS.max_paragraph_length,
      'max_sent_length': FLAGS.max_sent_length,
      'max_sent_per_paragraph': FLAGS.max_sent_per_paragraph,
  }


def main(unused_argv):
  tf.logging.info('GPU is available %s', tf.test.is_gpu_available())
  tf.random.set_random_seed(FLAGS.seed)
  # Other defaults moved to p2s_config.py for simplicity.
  # With TPUEstimator, batch_size is reserved in params.
  batch_size = FLAGS.batch_size

  # Determine model configuration, stored in params (except batch_size).
  # These include model hyper-parameters, and other graph construction settings.
  params = {}
  if FLAGS.params_file:
    with tf.gfile.Open(FLAGS.params_file, 'r') as f:
      params = json.load(f)
  else:
    tf.logging.error('Need to specify params file.')
  tie_encs = params['tie_sent_para_enc']
  tie_decs = params['tie_sent_para_dec']

  sptokens = [BOS, BOP, MASK] if tie_encs or tie_decs else [MASK]
  tk, spid_dict = util.get_tokenizer_with_special(FLAGS.vocab_file, sptokens)

  if not tk:
    tf.logging.fatal('failed to read vocab')

  params.update({'vocab_size': tk.vocab_size})
  if params['d_hidden_eq_hidden']:
    params.update({'d_hidden': (params['rnn_hidden_size']
                                if params['encoder_type'] == 'rnn'
                                else params['trf_hidden_size'])})

  # Write hyper parameters to json file in model directory.
  tf.logging.info('Hyper-parameters: %s', params)
  hypers_file = os.path.join(FLAGS.model_dir, 'hypers.json')
  if not tf.gfile.Exists(hypers_file):
    tf.gfile.MakeDirs(os.path.dirname(hypers_file))
    with tf.gfile.Open(hypers_file, 'w') as f:
      json.dump(params, f)

  run_config = tf.contrib.tpu.RunConfig(
      model_dir=FLAGS.model_dir,
      keep_checkpoint_max=0,
      save_checkpoints_steps=FLAGS.checkpoint_steps,
      master=FLAGS.master,
      tpu_config=tpu_config.TPUConfig(iterations_per_loop=1000,))
  ae_estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,  # If false, like a regular Estimator
      model_fn=get_model_fn(spid_dict),
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size,
      eval_on_tpu=False,   # In eval mode, avoid TPU weird APIs.
      params=params)

  if FLAGS.mode == 'train':
    do_augment = params['noisy_paragraph_prob'] > 0
    tf.logging.info('Train mode')
    if params['out_domain_pretrain_steps'] > 0:
      # Pre-train the model on out-of-domain data.
      ae_estimator.train(
          input_fn=get_input_fn(
              params,
              util.file_list(FLAGS.out_domain_data_dir, 'train',
                             FLAGS.out_domain_data_task),
              repeat=True, shuffle=True, roc_data_augment=do_augment),
          steps=params['out_domain_pretrain_steps'])
    steps_to_train = FLAGS.in_domain_pretrain_steps + FLAGS.train_steps
    ae_estimator.train(
        input_fn=get_input_fn(
            params,
            util.file_list(FLAGS.data_dir, 'train', FLAGS.task),
            repeat=True, shuffle=True,
            roc_data_augment=do_augment),
        steps=steps_to_train)

  elif FLAGS.mode in ('eval', 'decode', 'eval_and_decode'):
    assert FLAGS.eval_subset, 'Specify eval_subset.'
    subsets = six.ensure_str(FLAGS.eval_subset, 'utf-8').split(',')
    for s in subsets:
      assert s in ('train', 'valid', 'valid_gt', 'test', 'test_gt')
    ckpt_gen = util.checkpoint_file_gen(
        ae_estimator, FLAGS.eval_checkpoints, sleep_secs=30)
    for ckpt_file in ckpt_gen:
      global_step = ae_estimator.get_variable_value('global_step')

      for s in subsets:
        file_list = util.file_list(FLAGS.data_dir, s, FLAGS.task)
        have_ground_truth = s in ('valid_gt', 'test_gt')
        t0 = time.time()
        tf.logging.info('Run %s on %s with checkpoint %s', FLAGS.mode,
                        s, ckpt_file)
        if FLAGS.mode in ('eval', 'eval_and_decode'):

          ae_estimator.evaluate(
              input_fn=get_input_fn(params, file_list,
                                    repeat=False,
                                    # For eval, we depend on order of data.
                                    shuffle=False),
              steps=FLAGS.max_eval_steps,
              name=s,
              checkpoint_path=ckpt_file)
          tf.logging.info('Eval done in %g s', time.time() - t0)

        if FLAGS.mode in ('decode', 'eval_and_decode'):  # Decode
          output_dir = os.path.join(FLAGS.model_dir, FLAGS.decode_output_subdir)
          if have_ground_truth:
            assert len(file_list) == 1
            decode_eval = p2s_eval.P2sEval(util.get_seq_exs(file_list[0]))
          decode_summary_writer = tf.summary.FileWriter(os.path.join(
              output_dir, 'decode_eval_%s' % s))
          # Use whole dataset by not specifying max_steps
          decode_root_path = os.path.join(
              output_dir, 'decodes',
              '%s.%s' % (os.path.basename(ckpt_file), s))
          decode_file = decode_root_path + '.decode.txt'
          tf.logging.info('Write decodes to %s', decode_file)
          tf.gfile.MakeDirs(os.path.dirname(decode_file))
          def detok(s):
            return tk.decode(util.strip_after_eos(s))
          model_summaries = {}   # dict: First sent -> summary
          with tf.gfile.Open(decode_file, 'w') as f:
            for i, predictions in enumerate(ae_estimator.predict(
                input_fn=get_input_fn(params, file_list, repeat=False),
                checkpoint_path=ckpt_file)):

              summ = detok(predictions['summary'])

              if have_ground_truth:

                model_summaries[first_sentence(
                    predictions['first_sentence'])] = summ

              write_line(f, 'S|%s' % summ)
              opara = detok(predictions['document'])
              write_line(f, 'P|%s' % opara)
              if FLAGS.decode_reconstructions:
                write_line(f,
                           'R-P|%s' % detok(predictions['decoded_paragraph']))
                write_line(f,
                           'R-S|%s' % detok(predictions['decoded_sentences']))
                write_line(f,
                           'S-P|%s' % detok(predictions['expanded_paragraphs']))

              write_line(f, '----')
              if i == FLAGS.max_decodes:
                break

          tf.logging.info('Decode done in %g s', time.time() - t0)

          # Write decodes in canonical order (sorted by 1st sentence).
          # pylint: disable=unnecessary-lambda
          keys_sorted = sorted(model_summaries.keys(), key=lambda x: detok(x))
          with tf.gfile.Open(decode_root_path + '.decode_only', 'w') as f:
            for k in keys_sorted:
              f.write('%s\n' % model_summaries[k])

          if have_ground_truth:
            # Compute metrics vs ground-truth subset and write out
            # metrics file.
            with tf.gfile.Open(decode_root_path + '.decode_eval.csv', 'w') as f:
              metrics = decode_eval.compute_metrics(model_summaries)
              f.write(str(metrics))
            # Add to Tensorboard
            dec_summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag=x,
                    simple_value=metrics.metrics[x])
                for x in p2s_eval.Metrics.ALL_METRICS])
            decode_summary_writer.add_summary(dec_summary, global_step)
            decode_summary_writer.flush()
  else:
    tf.logging.fatal('Unexpected mode.')


if __name__ == '__main__':
  app.run(main)
