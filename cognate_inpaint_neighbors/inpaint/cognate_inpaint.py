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

"""Image pixel inpainting adapted for cognate reflex generation."""

import csv
import itertools
import json
import logging
import os
import random
import time
from typing import Sequence

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir',
                    'experimental/users/rws/st2022',
                    'Location of SIGTYP 2022 data.')
flags.DEFINE_string('data_division', '0.10',
                    'Which data division.')
flags.DEFINE_string('checkpoint_dir', None, 'Location of model checkpoint.')
flags.DEFINE_string('language_group', 'felekesemitic',
                    'Which language group.')
flags.DEFINE_integer('max_epochs', 100, 'Maximum number of epochs.')
flags.DEFINE_integer('steps_per_epoch', 1000, 'Steps per epoch.')
flags.DEFINE_integer('embedding_dim', 32, 'Character embedding size.')
flags.DEFINE_integer('kernel_width', 4, 'Kernel context size.')
flags.DEFINE_integer('filters', 128, 'Number of convolution filters.')
flags.DEFINE_enum('nonlinearity', 'leaky_relu', ['leaky_relu', 'relu', 'tanh'],
                  'Nonlinearity.')
flags.DEFINE_boolean('decode', False, 'Run evaluation against the test data.')
flags.DEFINE_string('output_results_tsv', None,
                    'Location of the output file with decoding results. The '
                    'file is in TSV format.')

_HPARAMS_FILENAME = 'hparams.json'  # Model configuration.
_VOCAB_FILENAME = 'vocab.txt'  # Symbol vocabulary.


class Infiller(tf.keras.Model):
  """The infiller convolutional model."""

  def __init__(self, vocab_size, hparams, batch_size, nlangs, max_length):
    super(Infiller, self).__init__()
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.embedding_dim = hparams['embedding_dim']
    self.units = hparams['filters']
    self.nlangs = nlangs
    self.max_length = max_length

    ##-------- Embedding layers in Encoder ------- ##
    self.embedding = tf.keras.layers.Embedding(vocab_size, self.embedding_dim)

    ##-------- Convolution ------- ##
    self.conv = tf.keras.layers.Conv2D(
        filters=self.units, kernel_size=(nlangs, hparams['kernel_width']))

    ##-------- Nonlinearity ------- ##
    if hparams['nonlinearity'] == 'leaky_relu':
      self.act = tf.keras.layers.LeakyReLU()
    elif hparams['nonlinearity'] == 'relu':
      self.act = tf.keras.layers.ReLU()
    else:
      self.act = tf.keras.layers.Activation('tanh')

    ##-------- Deconvolution ------- ##
    self.deconv = tf.keras.layers.Conv2DTranspose(
        filters=vocab_size, kernel_size=(nlangs, hparams['kernel_width']))

  def call(self, inputs, input_mask):
    # Reshape the mask.
    rmask = tf.repeat(input_mask, self.embedding_dim, axis=-1)
    rmask = tf.reshape(
        rmask,
        shape=(self.batch_size, self.nlangs, self.max_length,
               self.embedding_dim))

    # Embed the inputs.
    inputs = self.embedding(inputs)
    inputs = inputs * rmask

    # Convolve
    inputs = self.conv(inputs)
    inputs = self.act(inputs)

    # Deconvolve
    logits = self.deconv(inputs)

    return logits


def get_hparams(num_langs=0):
  """Builds hyper-parameter dictionary from flags or file."""
  if FLAGS.decode and FLAGS.checkpoint_dir:
    file_path = os.path.join(FLAGS.checkpoint_dir, _HPARAMS_FILENAME)
    if not os.path.isfile(file_path):
      raise FileNotFoundError(f'File {file_path} does not exist')
    logging.info('Reading hparams from %s ...', file_path)
    with open(file_path, 'r') as f:
      hparams = json.load(f)
      logging.info('HParams: %s', hparams)
      return hparams
  else:
    return {
        'embedding_dim': FLAGS.embedding_dim,
        'kernel_width': FLAGS.kernel_width,
        'filters': FLAGS.filters,
        'nonlinearity': FLAGS.nonlinearity,
        'num_langs': num_langs,
    }


@tf.function
def loss_function(real, pred, mask):
  # real shape = (BATCH_SIZE, max_length_output)
  # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size)
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')
  loss = cross_entropy(y_true=real, y_pred=pred)
  loss = mask * loss
  loss = tf.reduce_sum(loss)
  return loss


@tf.function
def train_step(infiller, optimizer, inp, inp_mask, targ, targ_mask):
  """Single training step."""
  loss = 0
  with tf.GradientTape() as tape:
    logits = infiller(inp, inp_mask)
    loss = loss_function(targ, logits, targ_mask)

  variables = infiller.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss


def evaluate_cset(infiller, cset, char2idx, max_length):
  """Evaluates given cognate set."""
  tgt_index = 0
  inputs = []
  input_mask = []
  # Find possible target positions.
  for i, p in enumerate(cset):
    if p.strip():
      if p == '<TARGET>':
        tgt_index = i
        inputs.append([char2idx['<TARGET>']] * max_length)
        input_mask.append([0.0] * max_length)
      else:
        seq = [char2idx['<BOS>']] + [
            char2idx[c] if c in char2idx else char2idx['<UNK>']
            for c in p.split()
        ] + [char2idx['<EOS>']]
        template = [char2idx['<EOS>']] * max_length
        for j in range(min(len(seq), max_length)):
          template[j] = seq[j]
        inputs.append(template)
        input_mask.append([1.0] * max_length)
    else:
      inputs.append([char2idx['<BLANK>']] * max_length)
      input_mask.append([0.0] * max_length)

  inputs = tf.constant([inputs], dtype='int32')
  input_mask = tf.constant([input_mask], dtype='float32')

  logits = infiller(inputs, input_mask)
  trow = tf.math.argmax(logits[0, tgt_index, :, :], axis=-1)
  return trow.numpy()


def silent_translate(infiller, cset, char2idx, max_length, idx2char):
  result = evaluate_cset(infiller, cset, char2idx, max_length)
  result = list(result)
  if char2idx['<EOS>'] in result:
    result = result[1:result.index(char2idx['<EOS>'])]
  else:
    result = result[1:]
  result = ' '.join([idx2char[x] for x in result])
  return result


def build_train_dataset(all_samples, batch_size, nlangs, max_length, char2idx):
  """Creates train dataset from the generator."""
  # Create data generators to feed into the networks.
  def la_gen():
    while True:
      for icset, tcset in all_samples:
        assert len(icset) == nlangs
        assert len(tcset) == nlangs
        inputs = []
        targets = []
        input_mask = []
        target_mask = []
        # Create the actual data content.
        for i in range(len(icset)):
          if icset[i] == '<TARGET>':
            seq = [char2idx['<BOS>']] + [
                char2idx[c] if c in char2idx else char2idx['<UNK>']
                for c in tcset[i].split()
            ] + [char2idx['<EOS>']]
            template = [char2idx['<EOS>']] * max_length
            for j in range(min(len(seq), max_length)):
              template[j] = seq[j]
            inputs.append([char2idx['<TARGET>']] * max_length)
            targets.append(template)
            input_mask.append([0.0] * max_length)
            target_mask.append([1.0] * max_length)
          elif icset[i] == '<BLANK>':
            inputs.append([char2idx['<BLANK>']] * max_length)
            targets.append([char2idx['<BLANK>']] * max_length)
            input_mask.append([0.0] * max_length)
            target_mask.append([0.0] * max_length)
          else:
            seq = [char2idx['<BOS>']] + [
                char2idx[c] if c in char2idx else char2idx['<UNK>']
                for c in icset[i].split()
            ] + [char2idx['<EOS>']]
            template = [char2idx['<EOS>']] * max_length
            for j in range(min(len(seq), max_length)):
              template[j] = seq[j]
            inputs.append(template)
            targets.append(template)
            input_mask.append([1.0] * max_length)
            target_mask.append([1.0] * max_length)
          # Convert to required tensor formats.
        inputs = tf.constant([inputs], dtype='int32')
        targets = tf.constant([targets], dtype='int32')
        input_mask = tf.constant([input_mask], dtype='float32')
        target_mask = tf.constant([target_mask], dtype='float32')

        yield (inputs, targets, input_mask, target_mask)

  return tf.data.Dataset.from_generator(
      la_gen,
      output_signature=(tf.TensorSpec(
          shape=(batch_size, nlangs, max_length), dtype='int32'),
                        tf.TensorSpec(
                            shape=(batch_size, nlangs, max_length),
                            dtype='int32'),
                        tf.TensorSpec(
                            shape=(batch_size, nlangs, max_length),
                            dtype='float32'),
                        tf.TensorSpec(
                            shape=(batch_size, nlangs, max_length),
                            dtype='float32')))


def expand_training_set(cogsets):
  """Expands the dataset to all possible variations."""
  logging.info('Expanding training data ...')
  nlangs = len(cogsets[0])
  all_samples = []
  for cs in cogsets:
    # Loop over all possible target positions.
    for i in range(nlangs):
      if cs[i]:
        # Find all remaining valid indexes...
        valids = []
        for j in range(nlangs):
          if j != i and cs[j]:
            valids.append(j)
        # Loop over all combinations of valids of size at least 1:
        for j in range(1, nlangs):
          for combo in itertools.combinations(valids, j):
            isample = []
            tsample = []
            for k, p in enumerate(cs):
              if k == i:
                isample.append('<TARGET>')
                tsample.append(p)
              elif k in combo:
                isample.append(p)
                tsample.append(p)
              else:
                isample.append('<BLANK>')
                tsample.append('<BLANK>')
            all_samples.append((isample, tsample))
  random.shuffle(all_samples)
  return all_samples


def build_base_data_and_vocab(datadir):
  """Builds vocabulary from the training data files."""
  vocab = set()
  cogsets = []
  filepath = os.path.join(datadir, f'training-{FLAGS.data_division}.tsv')
  logging.info('Preparing base training data from %s ...', filepath)
  with open(filepath, 'r', encoding='utf8') as fin:
    # Skip header.
    next(fin)
    for line in fin:
      parts = tuple(line.strip('\n').split('\t')[1:])
      for p in parts:
        for c in p.split():
          vocab.add(c)
      cogsets.append([p.strip() for p in parts])
  vocab = ['<PAD>', '<EOS>', '<BOS>', '<UNK>', '<TARGET>', '<BLANK>'
          ] + list(vocab)
  return cogsets, vocab


def get_vocab(checkpoint_dir):
  """Loads vocabulary from a file."""
  file_path = os.path.join(checkpoint_dir, _VOCAB_FILENAME)
  if not os.path.isfile(file_path):
    raise FileNotFoundError(f'File {file_path} does not exist')
  logging.info('Loading vocab from %s ...', file_path)
  with open(file_path, 'r', encoding='utf8') as f:
    vocab = [symbol.strip() for symbol in f if symbol]
  logging.info('%d symbols loaded.', len(vocab))
  return vocab


def load_test_set(datadir):
  """Loads test set from the files."""
  test_sets = []
  targets = []
  filepath = os.path.join(datadir, f'test-{FLAGS.data_division}.tsv')
  logging.info('Preparing test data from %s ...', filepath)
  with open(filepath, 'r', encoding='utf8') as fin:
    # Parse header.
    header = next(fin)
    # Parse the actual test data.
    for line in fin:
      tokens = line.strip('\n').split('\t')
      parts = []
      target_col = -1
      for i, p in enumerate(tuple(tokens[1:])):
        part = '<TARGET>' if p == '?' else p
        parts.append(part)
        if part == '<TARGET>':
          if target_col != -1:
            raise ValueError(f'{tokens[0]}: Multiple targets in the same set!')
          target_col = i
          targets.append((tokens[0], target_col))
      test_sets.append(parts)
  assert len(test_sets) == len(targets)
  return test_sets, header.split(), targets


def get_datadir():
  return os.path.join(FLAGS.data_dir, 'data', FLAGS.language_group)


def train_model(checkpoint_dir):
  """Training pipeline."""
  # Produce base training data and vocab, and expand the training data.
  datadir = get_datadir()
  cogsets, vocab = build_base_data_and_vocab(datadir)
  char2idx = {vocab[i]: i for i in range(len(vocab))}
  idx2char = {i: vocab[i] for i in range(len(vocab))}
  nlangs = len(cogsets[0])
  hparams = get_hparams(nlangs)
  vocab_size = len(vocab)
  all_samples = expand_training_set(cogsets)

  # Read in the test data.
  test_sets, _, _ = load_test_set(datadir)

  # Read in the baseline results.
  baseline_results = []
  with open(os.path.join(datadir, f'result-{FLAGS.data_division}.tsv'), 'r',
            encoding='utf8') as fin:
    # Skip header.
    next(fin)
    for line in fin:
      parts = tuple(line.strip('\n').split('\t')[1:])
      baseline_results.append(''.join(parts).strip())

  # Read in the solution set.
  solutions = []
  with open(os.path.join(datadir, f'solutions-{FLAGS.data_division}.tsv'), 'r',
            encoding='utf8') as fin:
    # Skip header.
    next(fin)
    for line in fin:
      parts = tuple(line.strip('\n').split('\t')[1:])
      solutions.append(''.join(parts).strip())

  # Core settings.
  steps_per_epoch = FLAGS.steps_per_epoch
  batch_size = 1
  max_length = 20
  # Have we written the vocab already?
  vocab_written = False

  # How many test samples per language?
  offset = len(solutions) / nlangs

  # Define the model, optimizer and loss function.
  infiller = Infiller(vocab_size, hparams, batch_size, nlangs, max_length)
  optimizer = tf.keras.optimizers.Adam()

  if checkpoint_dir:
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, infiller=infiller)

  # Train the model.
  best_error = None
  epochs = FLAGS.max_epochs

  logging.info('Training the model ...')
  train_dataset = build_train_dataset(all_samples, batch_size, nlangs,
                                      max_length, char2idx)
  for epoch in range(epochs):
    start = time.time()
    total_loss = 0
    for (_, (inp, targ, inp_mask,
             targ_mask)) in enumerate(train_dataset.take(steps_per_epoch)):
      batch_loss = train_step(infiller, optimizer, inp, inp_mask, targ,
                              targ_mask)
      total_loss += batch_loss

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # Evaluate on dev set:
    # Get model predictions.
    predictions = []
    for tset in test_sets:
      assert len(tset) == nlangs
      predictions.append(silent_translate(infiller, tset, char2idx, max_length,
                                          idx2char))
    # Calculate model error.
    terrors = []
    for lang in range(1, nlangs + 1):
      errors = 0
      merrors = 0
      for i, (_, _) in enumerate(zip(solutions, baseline_results)):
        if i >= (lang - 1) * offset and i < lang * offset:
          if solutions[i] != baseline_results[i]:
            errors += 1
          if solutions[i] != predictions[i]:
            merrors += 1
      print(lang, 'MODEL ERROR:', merrors / offset)
      terrors.append(merrors / offset)
    mean_accuracy = np.mean(terrors)
    print(mean_accuracy)
    if not best_error or mean_accuracy <= best_error:
      print('ERROR_UPDATE:', terrors)
      if checkpoint_dir:
        checkpoint.save(file_prefix=checkpoint_prefix)
        # Write the vocab and hparams AFTER ensuring checkpoint dir has been
        # created.
        if not vocab_written:
          with open(os.path.join(checkpoint_dir, _VOCAB_FILENAME), 'w',
                    encoding='utf8') as vfile:
            for v in vocab:
              vfile.write(v + '\n')
          vocab_written = True
          with open(os.path.join(checkpoint_dir, _HPARAMS_FILENAME),
                    'w') as hfile:
            json.dump(hparams, hfile)
      best_error = mean_accuracy
    print(best_error, '\n')

  # For some reason this step takes a couple of minutes to complete using
  # Tensorflow 2.8.0.
  logging.info('Done. Shutting down ...')


def decode_with_model(checkpoint_dir):
  """Runs the decoding on the unseen data."""
  if not FLAGS.output_results_tsv:
    raise app.UsageError('Specify --output_results_tsv')

  # Fetch the hyper-parameters and the vocab.
  hparams = get_hparams()
  assert 'num_langs' in hparams
  vocab = get_vocab(checkpoint_dir)
  char2idx = {vocab[i]: i for i in range(len(vocab))}
  idx2char = {i: vocab[i] for i in range(len(vocab))}
  vocab_size = len(vocab)
  batch_size = 1
  max_length = 20
  nlangs = hparams['num_langs']

  # Load the model from the latest checkpoint.
  latest_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
  if not latest_ckpt_path:
    raise ValueError('No checkpoint available')
  logging.info('Restoring from checkpoint %s ...', latest_ckpt_path)
  infiller = Infiller(vocab_size, hparams, batch_size, nlangs, max_length)
  checkpoint = tf.train.Checkpoint(infiller=infiller)
  checkpoint.restore(latest_ckpt_path).expect_partial()

  # Get model predictions.
  test_sets, columns, targets = load_test_set(get_datadir())
  logging.info('Generating predictions ...')
  predictions = []
  for i, test_set in enumerate(test_sets):
    assert len(test_set) == nlangs
    prediction = silent_translate(infiller, test_set, char2idx, max_length,
                                  idx2char)
    predictions.append((targets[i][0], targets[i][1], prediction))

  # Save the predictions.
  logging.info('Saving %d predictions to %s ...', len(predictions),
               FLAGS.output_results_tsv)
  with open(FLAGS.output_results_tsv, 'w', encoding='utf8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(columns)
    for cog_id, col_id, prediction in predictions:
      row = [cog_id]
      for i in range(len(columns)-1):
        if i == col_id:
          row.append(prediction)
        else:
          row.append(None)
      writer.writerow(row)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  checkpoint_dir = FLAGS.checkpoint_dir
  if not checkpoint_dir:
    raise app.UsageError('Specify --checkpoint_dir!')
  is_training = not FLAGS.decode
  if is_training:
    train_model(checkpoint_dir)
  else:
    decode_with_model(checkpoint_dir)


if __name__ == '__main__':
  app.run(main)
