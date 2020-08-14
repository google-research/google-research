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
r"""Script for training a captioning model."""
# pylint: disable=wildcard-import
# pylint: disable=unused-variable
# pylint: disable=undefined-variable
# pylint: disable=g-import-not-at-top
from __future__ import absolute_import
from __future__ import division

import pickle
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf

from hal.labeler.labeler_utils import *
from hal.learner.language_utils import pad_to_max_length
import hal.utils.word_vectorization as wv

if 'gfile' not in sys.modules:
  import tf.io.gfile as gfile

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', None, 'directory for saving models')

_SYNTHETIC_TRANSITION_PATH = None
_SYNTHETIC_TRANSITION_STATE_PATH = None
_SYNTHETIC_TRANSITION_LABEL_PATH = None
_TRANSITION_PATH = None
_TRANSITION_STATE_PATH = None
transition_label_path = None
vocab_path = None


def main(_):
  tf.enable_v2_behavior()
  ##############################################################################
  ######################### Data loading and processing ########################
  ##############################################################################
  print('Loading data')

  with gfile.GFile(_TRANSITION_PATH, 'r') as f:
    transitions = np.load(f)
  if np.max(transitions) > 1.0:
    transitions = transitions / 255.0
  with gfile.GFile(_SYNTHETIC_TRANSITION_PATH, 'r') as f:
    synthetic_tran_sitions = np.load(f)
  if np.max(synthetic_transitions) > 1.0:
    synthetic_transitions = synthetic_transitions / 255.0

  with gfile.GFile(transition_label_path, 'r') as f:
    captions = pickle.load(f)
  with gfile.GFile(_SYNTHETIC_TRANSITION_LABEL_PATH, 'r') as f:
    synthetic_captions = pickle.load(f)

  with gfile.GFile(vocab_path, 'r') as f:
    vocab_list = f.readlines()

  vocab_list = [w[:-1].decode('utf-8') for w in vocab_list]
  vocab_list = ['eos', 'sos'] + vocab_list

  v2i, i2v = wv.create_look_up_table(vocab_list)
  encode_fn = wv.encode_text_with_lookup_table(v2i)
  decode_fn = wv.decode_with_lookup_table(i2v)

  encoded_captions = []
  for all_cp in captions:
    for cp in all_cp:
      cp = 'sos ' + cp + ' eos'
      encoded_captions.append(np.array(encode_fn(cp)))

  synthetic_encoded_captions = []
  for all_cp in synthetic_captions:
    for cp in all_cp:
      cp = 'sos ' + cp + ' eos'
      synthetic_encoded_captions.append(np.array(encode_fn(cp)))

  all_caption_n = len(encoded_captions)
  all_synthetic_caption_n = len(synthetic_encoded_captions)

  encoded_captions = np.array(encoded_captions)
  encoded_captions = pad_to_max_length(encoded_captions, max_l=15)

  synthetic_encoded_captions = np.array(synthetic_encoded_captions)
  synthetic_encoded_captions = pad_to_max_length(
      synthetic_encoded_captions, max_l=15)

  obs_idx, caption_idx = [], []
  curr_caption_idx = 0
  for i, _ in enumerate(transitions):
    for cp in captions[i]:
      obs_idx.append(i)
      caption_idx.append(curr_caption_idx)
      curr_caption_idx += 1
  assert curr_caption_idx == all_caption_n

  synthetic_obs_idx, synthetic_caption_idx = [], []
  curr_caption_idx = 0
  for i, _ in enumerate(synthetic_transitions):
    for cp in synthetic_captions[i]:
      synthetic_obs_idx.append(i)
      synthetic_caption_idx.append(curr_caption_idx)
      curr_caption_idx += 1
  assert curr_caption_idx == all_synthetic_caption_n

  obs_idx = np.array(obs_idx)
  caption_idx = np.array(caption_idx)
  all_idx = np.arange(len(caption_idx))
  train_idx = all_idx[:int(len(all_idx) * 0.8)]
  test_idx = all_idx[int(len(all_idx) * 0.8):]
  print('Number of training examples: {}'.format(len(train_idx)))
  print('Number of test examples: {}\n'.format(len(test_idx)))

  synthetic_obs_idx = np.array(synthetic_obs_idx)
  synthetic_caption_idx = np.array(synthetic_caption_idx)
  synthetic_all_idx = np.arange(len(synthetic_caption_idx))
  synthetic_train_idx = synthetic_all_idx[:int(len(synthetic_all_idx) * 0.8)]
  synthetic_test_idx = synthetic_all_idx[int(len(synthetic_all_idx) * 0.8):]
  print('Number of synthetic training examples: {}'.format(
      len(synthetic_train_idx)))
  print('Number of synthetic test examples: {}\n'.format(
      len(synthetic_test_idx)))

  ##############################################################################
  ############################# Training Setup #################################
  ##############################################################################
  embedding_dim = 32
  units = 64
  vocab_size = len(vocab_list)
  batch_size = 64
  max_sequence_length = 15

  encoder_config = {'name': 'image', 'embedding_dim': 32}
  decoder_config = {
      'name': 'attention',
      'word_embedding_dim': 64,
      'hidden_units': 256,
      'vocab_size': len(vocab_list),
  }

  encoder = get_captioning_encoder(encoder_config)
  decoder = get_captioning_decoder(decoder_config)

  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')

  def loss_function(real, pred, sos_symbol=1):
    mask = tf.math.logical_not(tf.math.equal(real, sos_symbol))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

  @tf.function
  def train_step(input_tensor, target):
    """Traing on a batch of data."""
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([1] * target.shape[0], 1)

    with tf.GradientTape() as tape:
      features = encoder(input_tensor, training=True)
      for i in range(1, target.shape[1]):
        # passing the features through the decoder
        predictions, hidden, _ = decoder(
            dec_input, features, hidden, training=True)
        loss += loss_function(target[:, i], predictions)
        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

  @tf.function
  def evaluate_batch(input_tensor, target):
    """Evaluate loss on a batch of data."""
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([1] * target.shape[0], 1)
    features = encoder(input_tensor, training=False)

    for i in range(1, target.shape[1]):
      # passing the features through the decoder
      predictions, hidden, _ = decoder(
          dec_input, features, hidden, training=False)
      loss += loss_function(target[:, i], predictions)
      # using teacher forcing
      dec_input = tf.expand_dims(target[:, i], 1)
    total_loss = (loss / int(target.shape[1]))
    return total_loss

  ##############################################################################
  ############################# Training Loop ##################################
  ##############################################################################
  print('Start training...\n')
  start_epoch = 0
  if FLAGS.save_dir:
    checkpoint_path = FLAGS.save_dir
    ckpt = tf.train.Checkpoint(
        encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
      start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

  epochs = 400
  step_per_epoch = int(len(captions) / batch_size) * 10

  previous_best = 100.

  mixing_ratio = 0.4
  syn_bs = int(batch_size * 2 * mixing_ratio)
  true_bs = int(batch_size * 2 * (1 - mixing_ratio))

  for epoch in range(start_epoch, epochs):
    start = time.time()
    total_loss = 0

    for batch in range(step_per_epoch):
      batch_idx = np.random.choice(train_idx, size=true_bs)
      synthetic_batch_idx = np.random.choice(synthetic_train_idx, size=syn_bs)
      input_tensor = transitions[obs_idx[batch_idx], :]
      synthetic_input_tensor = synthetic_transitions[
          synthetic_obs_idx[synthetic_batch_idx], :]
      input_tensor = np.concatenate([input_tensor, synthetic_input_tensor],
                                    axis=0)
      input_tensor = encoder.preprocess(input_tensor)
      target = encoded_captions[caption_idx[batch_idx]]
      sythetic_target = synthetic_encoded_captions[
          synthetic_caption_idx[synthetic_batch_idx]]
      target = np.concatenate([target, sythetic_target], axis=0)
      batch_loss, t_loss = train_step(input_tensor, target)
      total_loss += t_loss

      if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(
            epoch + 1, batch,
            batch_loss.numpy() / int(target.shape[1])))

    if epoch % 5 == 0 and FLAGS.save_dir:
      test_total_loss = 0
      for batch in range(3):
        batch_idx = np.clip(np.arange(true_bs) + batch * true_bs, 0, 196)
        idx = test_idx[batch_idx]
        input_tensor = transitions[obs_idx[idx], :]
        target = encoded_captions[caption_idx[idx]]
        t_loss = evaluate_batch(input_tensor, target)
        test_total_loss += t_loss
        batch_idx = np.arange(syn_bs) + batch * syn_bs
        idx = synthetic_test_idx[batch_idx]
        input_tensor = synthetic_transitions[synthetic_obs_idx[idx], :]
        target = synthetic_encoded_captions[synthetic_caption_idx[idx]]
        t_loss = evaluate_batch(input_tensor, target)
        test_total_loss += t_loss
      test_total_loss /= 6.
      if test_total_loss < previous_best:
        previous_best = test_total_loss
        ckpt_manager.save(checkpoint_number=epoch)

    print('Epoch {} | Loss {:.6f} | Val loss {:.6f}'.format(
        epoch + 1, total_loss / step_per_epoch, previous_best))
    print('Time taken for 1 epoch {:.6f} sec\n'.format(time.time() - start))

    if epoch % 20 == 0:
      total_loss = 0
      for batch in range(len(test_idx) // batch_size):
        batch_idx = np.arange(batch_size) + batch * batch_size
        idx = test_idx[batch_idx]
        input_tensor = transitions[obs_idx[idx], :]
        target = encoded_captions[caption_idx[idx]]
        # input_tensor = input_tensor[:, 0] - input_tensor[:, 1]
        t_loss = evaluate_batch(input_tensor, target)
        total_loss += t_loss

      print('====================================================')
      print('Test Loss {:.6f}'.format(total_loss /
                                      (len(test_idx) // batch_size)))
      print('====================================================\n')


if __name__ == '__main__':
  app.run(main)
