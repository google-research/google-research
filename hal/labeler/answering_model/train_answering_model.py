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
  # with gfile.GFile(transition_path, 'r') as f:
  #   transitions = np.load(f)

  with gfile.GFile(transition_state_path, 'r') as f:
    states = np.load(f)
  states = np.float32(states)

  with gfile.GFile(transition_label_path, 'r') as f:
    captions = pickle.load(f)

  with gfile.GFile(answer_path, 'r') as f:
    answers = pickle.load(f)

  with gfile.GFile(vocab_path, 'r') as f:
    vocab_list = f.readlines()

  vocab_list = [w[:-1].decode('utf-8') for w in vocab_list]
  vocab_list = ['eos', 'sos', 'nothing'] + vocab_list
  vocab_list[-1] = 'to'

  v2i, i2v = wv.create_look_up_table(vocab_list)
  encode_fn = wv.encode_text_with_lookup_table(v2i)
  decode_fn = wv.decode_with_lookup_table(i2v)

  caption_decoding_map = {v: k for k, v in captions[0].items()}
  decompressed_captions = []
  for caption in captions[1:]:
    new_caption = []
    for c in caption:
      new_caption.append(caption_decoding_map[c])
    decompressed_captions.append(new_caption)
  captions = decompressed_captions

  encoded_captions = []
  new_answers = []
  for i, all_cp in enumerate(captions):
    for cp in all_cp:
      encoded_captions.append(np.array(encode_fn(cp)))
    for a in answers[i]:
      new_answers.append(float(a))
  all_caption_n = len(encoded_captions)
  encoded_captions = np.array(encoded_captions)
  encoded_captions = pad_to_max_length(encoded_captions)
  answers = np.float32(new_answers)

  obs_idx, caption_idx = [], []
  curr_caption_idx = 0
  for i, _ in enumerate(states):
    for cp in captions[i]:
      obs_idx.append(i)
      caption_idx.append(curr_caption_idx)
      curr_caption_idx += 1
  assert curr_caption_idx == all_caption_n

  obs_idx = np.array(obs_idx)
  caption_idx = np.array(caption_idx)
  all_idx = np.arange(len(caption_idx))
  train_idx = all_idx[:int(len(all_idx) * 0.7)]
  test_idx = all_idx[int(len(all_idx) * 0.7):]
  print('Number of training examples: {}'.format(len(train_idx)))
  print('Number of test examples: {}\n'.format(len(test_idx)))

  ##############################################################################
  ############################# Training Setup #################################
  ##############################################################################
  embedding_dim = 32
  units = 64
  vocab_size = len(vocab_list)
  batch_size = 128
  max_sequence_length = 21

  encoder_config = {'name': 'state', 'embedding_dim': 64}
  decoder_config = {
      'name': 'state',
      'word_embedding_dim': 64,
      'hidden_units': 512,
      'vocab_size': len(vocab_list),
  }

  encoder = get_answering_encoder(encoder_config)
  decoder = get_answering_decoder(decoder_config)
  projection_layer = tf.keras.layers.Dense(
      1, activation='sigmoid', name='answering_projection')

  optimizer = tf.keras.optimizers.Adam(1e-4)
  bce = tf.keras.losses.BinaryCrossentropy()

  @tf.function
  def compute_loss(obs, instruction, target):
    instruction = tf.expand_dims(instruction, axis=-1)
    hidden = decoder.reset_state(batch_size=target.shape[0])
    features = encoder(obs)
    for i in tf.range(max_sequence_length):
      _, hidden, _ = decoder(instruction[:, i], features, hidden)
    projection = tf.squeeze(projection_layer(hidden), axis=1)
    loss = bce(target, projection)
    return loss, projection

  @tf.function
  def train_step(obs, instruction, target):
    with tf.GradientTape() as tape:
      loss, _ = compute_loss(obs, instruction, target)
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables + projection_layer.trainable_variables
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

  for epoch in range(start_epoch, epochs):
    start = time.time()
    total_loss = 0
    for batch in range(step_per_epoch):
      batch_idx = np.random.choice(train_idx, size=batch_size)
      input_tensor = tf.convert_to_tensor(states[obs_idx[batch_idx], :])
      instruction = tf.convert_to_tensor(
          encoded_captions[caption_idx[batch_idx]])
      target = tf.convert_to_tensor(answers[caption_idx[batch_idx]])
      batch_loss = train_step(input_tensor, instruction, target)
      total_loss += batch_loss

      if batch % 1000 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch,
                                                     batch_loss.numpy()))

    if epoch % 5 == 0 and FLAGS.save_dir:
      test_total_loss = 0
      accuracy = 0
      for batch in range(10):
        batch_idx = np.arange(batch_size) + batch * batch_size
        idx = test_idx[batch_idx]
        input_tensor = tf.convert_to_tensor(states[obs_idx[idx], :])
        instruction = tf.convert_to_tensor(encoded_captions[caption_idx[idx]])
        target = tf.convert_to_tensor(answers[caption_idx[idx]])
        t_loss, prediction = compute_loss(input_tensor, instruction, target)
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
        batch_idx = np.arange(batch_size) + batch * batch_size
        idx = test_idx[batch_idx]
        input_tensor = tf.convert_to_tensor(states[obs_idx[idx], :])
        instruction = tf.convert_to_tensor(encoded_captions[caption_idx[idx]])
        target = tf.convert_to_tensor(answers[caption_idx[idx]])
        t_loss, prediction = compute_loss(input_tensor, instruction, target)
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
