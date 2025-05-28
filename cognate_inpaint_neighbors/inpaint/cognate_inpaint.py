# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
flags.DEFINE_string('train_file', 'training-mod_0.10.tsv',
                    'Location of SIGTYP 2022 training data within data_dir.')
flags.DEFINE_string('dev_file', 'dev_0.10.tsv',
                    'Location of SIGTYP 2022 dev data within data_dir.')
flags.DEFINE_string(
    'dev_solutions_file', 'dev_solutions_0.10.tsv',
    'Location of SIGTYP 2022 dev solution data within data_dir.')
flags.DEFINE_string('test_file', 'test_0.10.tsv',
                    'Location of SIGTYP 2022 test data within data_dir.')
flags.DEFINE_string('preds_file', 'model_result_0.10.tsv',
                    'Name of the output file were to put decoding results'
                    '(in the data_dir). The file is in TSV format.')
flags.DEFINE_string('checkpoint_dir', None, 'Location of model checkpoint.')
flags.DEFINE_integer('max_epochs', 150, 'Maximum number of epochs.')
flags.DEFINE_integer('steps_per_epoch', 500, 'Steps per epoch.')
flags.DEFINE_integer('embedding_dim', 32, 'Character embedding size.')
flags.DEFINE_integer('kernel_width', 4, 'Kernel context size.')
flags.DEFINE_integer('filters', 128, 'Number of convolution filters.')
flags.DEFINE_float('dropout', 0.1, 'Number of convolution filters.')
flags.DEFINE_enum('nonlinearity', 'leaky_relu', ['leaky_relu', 'relu', 'tanh'],
                  'Nonlinearity.')
flags.DEFINE_enum('sfactor', 'inputs', ['inputs', 'conv', 'none'],
                  'Where to renormalize the convolution filters, if at all.')
flags.DEFINE_boolean('decode', False, 'Run evaluation against the test data.')


_HPARAMS_FILENAME = 'hparams.json'  # Model configuration.
_VOCAB_FILENAME = 'vocab.txt'  # Symbol vocabulary.


class Infiller(tf.keras.Model):
  """The infiller convolutional model."""

  def __init__(self, vocab_size, hparams, batch_size, nlangs, max_length):
    super(Infiller, self).__init__()
    self.batch_size = batch_size
    self.units = hparams['filters']
    self.kernel_width = hparams['kernel_width']
    self.vocab_size = vocab_size
    self.embedding_dim = hparams['embedding_dim']
    self.nlangs = nlangs
    self.max_length = max_length
    self.scale_pos = hparams['sfactor']

    ##-------- Embedding layers in Encoder ------- ##
    self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                               self.embedding_dim)

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

    ##-------- Dropout ------- ##
    self.dropout = tf.keras.layers.Dropout(hparams['dropout'])

    ##-------- Deconvolution ------- ##
    self.deconv = tf.keras.layers.Conv2DTranspose(
        filters=self.vocab_size, kernel_size=(nlangs, hparams['kernel_width']))

  def call(self, inputs, input_mask, training):
    # Reshape the mask.
    rmask = tf.repeat(input_mask, self.embedding_dim, axis=-1)
    rmask = tf.reshape(
        rmask,
        shape=(self.batch_size, self.nlangs, self.max_length,
               self.embedding_dim))

    # Embed the inputs.
    inputs = self.embedding(inputs)
    inputs = inputs * rmask

    # Scale the inputs.
    sfactor = (self.nlangs * self.max_length) / tf.math.reduce_sum(input_mask)
    if self.scale_pos == 'inputs':
      inputs = inputs * sfactor

    # Convolve
    inputs = self.conv(inputs)
    if self.scale_pos == 'conv':
      inputs = inputs * sfactor
    inputs = self.dropout(inputs, training=training)
    inputs = self.act(inputs)

    # Deconvolve
    logits = self.deconv(inputs)

    return logits


def get_hparams():
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
        'dropout': FLAGS.dropout,
        'sfactor': FLAGS.sfactor,
    }


@tf.function
def loss_function(real, pred, mask):
  # real shape = (BATCH_SIZE, max_length_output)
  # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
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
    logits = infiller(inp, inp_mask, training=True)
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
  # Find possible target positions
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
        template = [char2idx['<BLANK>']] * max_length
        for j in range(min(len(seq), max_length)):
          template[j] = seq[j]
        inputs.append(template)
        input_mask.append([1.0] * max_length)
    else:
      inputs.append([char2idx['<BLANK>']] * max_length)
      input_mask.append([0.0] * max_length)

  inputs = tf.constant([inputs], dtype='int32')
  input_mask = tf.constant([input_mask], dtype='float32')

  logits = infiller(inputs, input_mask, training=False)
  trow = tf.math.argmax(logits[0, tgt_index, :, :], axis=-1)
  return trow.numpy()


def silent_translate(infiller, cset, char2idx, max_length, idx2char):
  result = evaluate_cset(infiller, cset, char2idx, max_length)
  result = list(result)
  result = ' '.join([
      idx2char[x] for x in result if idx2char[x] not in
      ['<PAD>', '<EOS>', '<BOS>', '<UNK>', '<TARGET>', '<BLANK>']
  ])
  return result


def build_train_dataset(all_samples, batch_size, nlangs, max_length, char2idx):
  """Creates train dataset from the generator."""
  # Create data generators to feed into the networks.
  def la_gen():
    while True:
      for icset in all_samples:
        inputs = []
        targets = []
        input_mask = []
        target_mask = []
        # Get the present items.
        valids = [i for i in range(len(icset)) if icset[i] != '<BLANK>']
        # Select how many inputs will be present to provide information.
        num_present = random.randint(1, len(valids))
        present = random.sample(valids, num_present)
        # Create the actual data content.
        for i in range(len(icset)):
          # Create a max_length sequence.
          template = [char2idx['<BLANK>']] * max_length
          seq = [char2idx['<BOS>']] + [
              char2idx[c] if c in char2idx else char2idx['<UNK>']
              for c in icset[i].split()
          ] + [char2idx['<EOS>']]
          for j in range(min(len(seq), max_length)):
            template[j] = seq[j]
          targets.append(template)
          inputs.append(template)
          # If the sequence if valid, get gradient from it.
          if i in valids:
            target_mask.append([1.0] * max_length)
          else:
            target_mask.append([0.0] * max_length)
          # If the sequence should be present, don't mask it.
          if i in present:
            input_mask.append([1.0] * max_length)
          else:
            input_mask.append([0.0] * max_length)
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
    # Find all valid positions.
    isample = []
    for i in range(nlangs):
      if cs[i]:
        isample.append(cs[i])
      else:
        isample.append('<BLANK>')
    all_samples.append(isample)
  random.shuffle(all_samples)
  return all_samples


def build_base_data_and_vocab(datadir):
  """Builds vocabulary from the training data files."""
  vocab = set()
  cogsets = []
  filepath = os.path.join(datadir, FLAGS.train_file)
  logging.info('Preparing base training data from %s ...', filepath)
  with open(filepath, 'r', encoding='utf-8') as fin:
    # Skip header.
    next(fin)
    for line in fin:
      parts = tuple(line.strip('\n').split('\t')[1:])
      for p in parts:
        for c in p.split():
          vocab.add(c)
      cogsets.append([p.strip() for p in parts])
  vocab = ['<PAD>', '<EOS>', '<BOS>', '<UNK>', '<TARGET>', '<BLANK>'] + sorted(
      list(vocab))
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


def train_model(checkpoint_dir):
  """Training pipeline."""


  # Produce base training data and vocab, and expand the training data.
  datadir = FLAGS.data_dir
  cogsets, vocab = build_base_data_and_vocab(datadir)
  char2idx = {vocab[i]: i for i in range(len(vocab))}
  idx2char = {i: vocab[i] for i in range(len(vocab))}
  nlangs = len(cogsets[0])
  hparams = get_hparams()
  vocab_size = len(vocab)
  all_samples = expand_training_set(cogsets)

  # Read in the dev data.
  dev_sets = []
  filepath = os.path.join(datadir, FLAGS.dev_file)
  with open(filepath, 'r', encoding='utf-8') as fin:
    # Skip header.
    next(fin)
    for line in fin:
      parts = tuple(line.strip('\n').split('\t')[1:])
      parts = ['<TARGET>' if p == '?' else p for p in parts]
      dev_sets.append(parts)

  # Read in the dev  solution set.
  dev_solutions = []
  filepath = os.path.join(datadir, FLAGS.dev_solutions_file)
  with open(filepath, 'r', encoding='utf-8') as fin:
    # Skip header.
    next(fin)
    for line in fin:
      parts = tuple(line.strip('\n').split('\t')[1:])
      dev_solutions.append(''.join(parts).strip())

  # Core settings.
  steps_per_epoch = FLAGS.steps_per_epoch
  batch_size = 1
  max_length = 20
  # Have we written the vocab and hparams already?
  vocab_written = False

  # Define the model, optimizer and loss function.
  infiller = Infiller(vocab_size, hparams, batch_size, nlangs, max_length)
  optimizer = tf.keras.optimizers.Adam()

  if checkpoint_dir:
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, infiller=infiller)

  logging.info('Training the model ...')
  train_dataset = build_train_dataset(all_samples, batch_size, nlangs,
                                      max_length, char2idx)
  best_error = None

  for epoch in range(FLAGS.max_epochs):
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
    derrors = [0 for l in range(nlangs)]
    dtotals = [0 for l in range(nlangs)]
    allerrors = 0
    for dset, dsol in zip(dev_sets, dev_solutions):
      tgt_index = dset.index('<TARGET>')
      dtotals[tgt_index] += 1
      pred = silent_translate(infiller, dset, char2idx, max_length, idx2char)
      if pred != dsol:
        derrors[tgt_index] += 1
        allerrors += 1
    derrors = [x / y for x, y in zip(derrors, dtotals) if y != 0]
    mean_accuracy = np.mean(derrors)

    # Update based on dev set.
    if not best_error or mean_accuracy <= best_error:
      print('ERROR_UPDATE:', derrors)
      if checkpoint_dir:
        checkpoint.save(file_prefix=checkpoint_prefix)
        # Write the vocab AFTER ensuring checkpoint dir has been created.
        if not vocab_written:
          # Write the model parameters.
          hparams = {}
          hparams['embedding_dim'] = FLAGS.embedding_dim
          hparams['kernel_width'] = FLAGS.kernel_width
          hparams['filters'] = FLAGS.filters
          hparams['dropout'] = FLAGS.dropout
          hparams['nonlinearity'] = FLAGS.nonlinearity
          hparams['sfactor'] = FLAGS.sfactor
          with open(checkpoint_dir + '/hparams.json', 'w') as vfile:
            json.dump(hparams, vfile)
          # Write the vocabulary.
          with open(
              checkpoint_dir + '/vocab.txt', 'w', encoding='utf-8') as vfile:
            for v in vocab:
              vfile.write(v + '\n')
          vocab_written = True
      best_error = mean_accuracy
    print(best_error, mean_accuracy, '\n')

  # For some reason this step takes a couple of minutes to complete using
  # Tensorflow 2.8.0.
  logging.info('Done. Shutting down ...')


def decode_with_model(checkpoint_dir):
  """Runs the decoding on the unseen data."""
  if not FLAGS.preds_file:
    raise app.UsageError('Specify --preds_file')

  # Fetch the hyper-parameters and the vocab.
  hparams = get_hparams()
  vocab = get_vocab(checkpoint_dir)
  char2idx = {vocab[i]: i for i in range(len(vocab))}
  idx2char = {i: vocab[i] for i in range(len(vocab))}
  vocab_size = len(vocab)
  batch_size = 1
  max_length = 20

  test_filepath = os.path.join(FLAGS.data_dir, FLAGS.test_file)
  preds_filepath = os.path.join(FLAGS.data_dir, FLAGS.preds_file)

  # Infer nlangs from test file header
  with open(test_filepath, 'r', encoding='utf-8') as fin:
    nlangs = len(next(fin).strip('\n').split('\t')) - 1

  # Load the model from the latest checkpoint.
  latest_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
  if not latest_ckpt_path:
    raise ValueError('No checkpoint available')
  logging.info('Restoring from checkpoint %s ...', latest_ckpt_path)
  infiller = Infiller(vocab_size, hparams, batch_size, nlangs, max_length)
  checkpoint = tf.train.Checkpoint(infiller=infiller)
  checkpoint.restore(latest_ckpt_path).expect_partial()

  # Generate predictions.
  logging.info('Generating predictions and saving results...')
  with open(preds_filepath, 'w', encoding='utf-8') as vfile:
    with open(test_filepath, 'r', encoding='utf-8') as tfile:
      # Copy the header.
      vfile.write(next(tfile))
      for line in tfile:
        parts = line.strip('\n').split('\t')
        tset = ['<TARGET>' if p == '?' else p for p in parts[1:]]
        tgt_index = tset.index('<TARGET>')
        pred = silent_translate(infiller, tset, char2idx, max_length, idx2char)
        row = ['' for p in parts]
        row[0] = parts[0]
        row[tgt_index + 1] = pred
        vfile.write('\t'.join(row) + '\n')


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
