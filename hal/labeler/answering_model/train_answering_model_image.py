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

# Lint as: python3
r"""Script for training a captioning model."""
# pylint: disable=unused-variable
# pylint: disable=undefined-variable
# pylint: disable=wildcard-import
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

synthetic_transition_path = None
synthetic_transition_state_path = None
synthetic_transition_label_path = None
transition_path = None
transition_state_path = None
transition_label_path = None
vocab_path = None


def main(_):
  tf.enable_v2_behavior()
  ##############################################################################
  ######################### Data loading and processing ########################
  ##############################################################################
  print('Loading data')

  with gfile.GFile(transition_path, 'r') as f:
    transitions = np.load(f)
  if np.max(transitions) > 1.0:
    transitions = transitions / 255.0
  with gfile.GFile(synthetic_transition_path, 'r') as f:
    synthetic_transitions = np.load(f)
  if np.max(synthetic_transitions) > 1.0:
    synthetic_transitions = synthetic_transitions / 255.0

  with gfile.GFile(transition_label_path, 'r') as f:
    captions = pickle.load(f)
  with gfile.GFile(synthetic_transition_label_path, 'r') as f:
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

  obs_idx, caption_idx, negative_caption_idx = [], [], []
  curr_caption_idx = 0
  for i, _ in enumerate(transitions):
    for cp in captions[i]:
      obs_idx.append(i)
      if 'nothing' not in cp:
        caption_idx.append(curr_caption_idx)
      else:
        negative_caption_idx.append(curr_caption_idx)
      curr_caption_idx += 1
  assert curr_caption_idx == all_caption_n

  synthetic_obs_idx, synthetic_caption_idx = [], []
  synthetic_negative_caption_idx = []
  curr_caption_idx = 0
  for i, _ in enumerate(synthetic_transitions):
    for cp in synthetic_captions[i]:
      synthetic_obs_idx.append(i)
      if 'nothing' not in cp:
        synthetic_caption_idx.append(curr_caption_idx)
      else:
        synthetic_negative_caption_idx.append(curr_caption_idx)
      curr_caption_idx += 1
  assert curr_caption_idx == all_synthetic_caption_n

  obs_idx = np.array(obs_idx)
  caption_idx = np.array(caption_idx)
  negative_caption_idx = np.array(negative_caption_idx)
  all_idx = np.arange(len(caption_idx))
  train_idx = all_idx[:int(len(all_idx) * 0.8)]
  test_idx = all_idx[int(len(all_idx) * 0.8):]
  print('Number of training examples: {}'.format(len(train_idx)))
  print('Number of test examples: {}\n'.format(len(test_idx)))

  synthetic_obs_idx = np.array(synthetic_obs_idx)
  synthetic_caption_idx = np.array(synthetic_caption_idx)
  synthetic_negative_caption_idx = np.array(synthetic_negative_caption_idx)
  synthetic_all_idx = np.arange(len(synthetic_caption_idx))
  synthetic_train_idx = synthetic_all_idx[:int(len(synthetic_all_idx) * 0.8)]
  synthetic_test_idx = synthetic_all_idx[int(len(synthetic_all_idx) * 0.8):]
  print('Number of synthetic training examples: {}'.format(
      len(synthetic_train_idx)))
  print('Number of synthetic test examples: {}\n'.format(
      len(synthetic_test_idx)))

  def sample_batch(data_type, batch_size, mode='train'):
    is_synthetic = data_type == 'synthetic'
    transitions_s = synthetic_transitions if is_synthetic else transitions
    encoded_captions_s = synthetic_encoded_captions if is_synthetic else encoded_captions
    obs_idx_s = synthetic_obs_idx if is_synthetic else obs_idx
    caption_idx_s = synthetic_caption_idx if is_synthetic else caption_idx
    all_idx_s = synthetic_all_idx if is_synthetic else all_idx
    train_idx_s = synthetic_train_idx if is_synthetic else train_idx
    test_idx_s = synthetic_test_idx if is_synthetic else test_idx
    if mode == 'train':
      batch_idx_s = np.random.choice(train_idx_s, size=batch_size)
    else:
      batch_idx_s = np.random.choice(test_idx_s, size=batch_size)
    input_tensor = tf.convert_to_tensor(
        np.concatenate([
            transitions_s[obs_idx_s[batch_idx_s], 1, :],
            transitions_s[obs_idx_s[batch_idx_s], 1, :]
        ]))
    positive_idx = caption_idx_s[batch_idx_s]
    negative_idx = caption_idx_s[np.random.choice(train_idx_s, size=batch_size)]
    caption_tensor = tf.convert_to_tensor(
        np.concatenate([
            encoded_captions_s[positive_idx], encoded_captions_s[negative_idx]
        ], axis=0))
    target_tensor = tf.convert_to_tensor(
        np.float32(np.concatenate([np.ones(batch_size),
                                   np.zeros(batch_size)], axis=0)))
    return input_tensor, caption_tensor, target_tensor

  ##############################################################################
  ############################# Training Setup #################################
  ##############################################################################
  embedding_dim = 32
  units = 64
  vocab_size = len(vocab_list)
  batch_size = 64
  max_sequence_length = 15

  encoder_config = {'name': 'image', 'embedding_dim': 64}
  decoder_config = {
      'name': 'attention',
      'word_embedding_dim': 64,
      'hidden_units': 256,
      'vocab_size': len(vocab_list),
  }

  encoder = get_answering_encoder(encoder_config)
  decoder = get_answering_decoder(decoder_config)
  projection_layer = tf.keras.layers.Dense(
      1, activation='sigmoid', name='answering_projection')

  optimizer = tf.keras.optimizers.Adam(1e-4)
  bce = tf.keras.losses.BinaryCrossentropy()

  @tf.function
  def compute_loss(obs, instruction, target, training):
    print('Build compute loss...')
    instruction = tf.expand_dims(instruction, axis=-1)
    hidden = decoder.reset_state(batch_size=target.shape[0])
    features = encoder(obs, training=training)
    for i in tf.range(max_sequence_length):
      _, hidden, _ = decoder(
          instruction[:, i], features, hidden, training=training)
    projection = tf.squeeze(projection_layer(hidden), axis=1)
    loss = bce(target, projection)
    return loss, projection

  @tf.function
  def train_step(obs, instruction, target):
    print('Build train step...')
    with tf.GradientTape() as tape:
      loss, _ = compute_loss(obs, instruction, target, True)
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables + projection_layer.trainable_variables
    print('num trainable: ', len(trainable_variables))
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss

  ##############################################################################
  ############################# Training Loop ##################################
  ##############################################################################
  print('Start training...\n')
  start_epoch = 0
  if FLAGS.save_dir:
    checkpoint_path = FLAGS.save_dir
    ckpt = tf.train.Checkpoint(
        encoder=encoder,
        decoder=decoder,
        projection_layer=projection_layer,
        optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
      start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

  epochs = 400
  step_per_epoch = int(all_caption_n / batch_size)

  previous_best, previous_best_accuracy = 100., 0.0
  # input_tensor, instruction, target = sample_batch('synthetic', batch_size,
  #                                                  'train')
  for epoch in range(start_epoch, epochs):
    start = time.time()
    total_loss = 0
    for batch in range(step_per_epoch):
      input_tensor, instruction, target = sample_batch('synthetic', batch_size,
                                                       'train')
      batch_loss = train_step(input_tensor, instruction, target)
      total_loss += batch_loss
      # print(batch, batch_loss)
      # print(instruction[0])
      # print(encode_fn('nothing'))
      # print('====================================')

      if batch % 1000 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch,
                                                     batch_loss.numpy()))

    if epoch % 5 == 0 and FLAGS.save_dir:
      test_total_loss = 0
      accuracy = 0
      for batch in range(10):
        input_tensor, instruction, target = sample_batch(
            'synthetic', batch_size, 'test')
        t_loss, prediction = compute_loss(input_tensor, instruction, target,
                                          False)
        test_total_loss += t_loss
        accuracy += np.mean(np.float32(np.float32(prediction > 0.5) == target))
      test_total_loss /= 10.
      accuracy /= 10.
      if accuracy > previous_best_accuracy:
        previous_best_accuracy, previous_best = accuracy, test_total_loss
        ckpt_manager.save(checkpoint_number=epoch)

    print('\nEpoch {} | Loss {:.6f} | Val loss {:.6f} | Accuracy {:.3f}'.format(
        epoch + 1, total_loss / step_per_epoch, previous_best,
        previous_best_accuracy))
    print('Time taken for 1 epoch {:.6f} sec\n'.format(time.time() - start))

    if epoch % 10 == 0:
      test_total_loss = 0
      accuracy = 0
      for batch in range(len(test_idx) // batch_size):
        input_tensor, instruction, target = sample_batch(
            'synthetic', batch_size, 'test')
        t_loss, prediction = compute_loss(
            input_tensor, instruction, target, training=False)
        test_total_loss += t_loss
        accuracy += np.mean(np.float32(np.float32(prediction > 0.5) == target))
      test_total_loss /= (len(test_idx) // batch_size)
      accuracy /= (len(test_idx) // batch_size)
      if accuracy > previous_best_accuracy and FLAGS.save_dir:
        previous_best_accuracy, previous_best = accuracy, test_total_loss
        ckpt_manager.save(checkpoint_number=epoch)
      print('\n====================================================')
      print('Test Loss {:.6f} | Test Accuracy {:.3f}'.format(
          test_total_loss, accuracy))
      print('====================================================\n')


if __name__ == '__main__':
  app.run(main)
