# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Example of training the SLiMPerformer on PennTreeBank and Enwik8 data, as well as the copy task."""
import collections
import gzip
import os
import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import torch

from performer.models.pytorch.slim_performer import slim_performer_model

FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 10, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-4, 'Adam Optimizer learning rate.')
flags.DEFINE_integer(
    'step_size', -1,
    'Tradeoff between memory and parallel running time (C). -1 corresponds to naive FULL method in the paper.'
)
flags.DEFINE_integer('hidden_dim', 512, 'Feature dimension.')
flags.DEFINE_integer('n_layers', 6, 'Number of Attention layers.')
flags.DEFINE_integer('ffn_dim', 2048, 'MLP dimension in model.')
flags.DEFINE_integer('n_heads', 8, 'Number of heads for attention.')
flags.DEFINE_string(
    'feature_type', 'sqr',
    'Nonlinearity function for feature. Can be relu, elu+1, sqr, favor+, or favor+{int}.'
)
flags.DEFINE_enum(
    'compute_type', 'iter', ['iter', 'ps', 'parallel_ps'],
    'Which type of method to compute: iter = iterative algorithm from Appendix B, ps = implementation using torch.cumsum, parallel_ps = implementation using custom log prefix sum implementation.'
)
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay for regularization.')

# Training parameters
flags.DEFINE_integer('iters_count', 100000, 'Number of training iterations.')
flags.DEFINE_bool('finetune', True, '')
flags.DEFINE_bool('on_gptln', True, 'Use layer norm after attention or before.')
flags.DEFINE_string('gpu_id', '0', 'ID of GPU.')
flags.DEFINE_integer(
    'arg_code', -1,
    'If -1, uses user-defined FLAGS. Else uses predetermined flag values to reproduce paper results.'
)
flags.DEFINE_integer('random_seed', 42, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('val_step', 5, 'Interval to predict validation metrics.')
flags.DEFINE_integer('print_step', 100, 'Interval to print metrics.')

# Dataset parameters
flags.DEFINE_enum('dataset', 'ptb', ['ptb', 'enwik8', 'copy'],
                  'Dataset to use.')
flags.DEFINE_integer('seq_len', 512, 'Maximum sequence length (L).')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')


def get_batch(data, batch_size, seq_len, index):
  """Batch the data."""
  elems_in_batch = batch_size * seq_len
  batches_count = len(data) // elems_in_batch

  batch_start = elems_in_batch * (index % batches_count)
  batch_end = batch_start + elems_in_batch

  batch = data[batch_start:batch_end]
  batch = batch.reshape(batch_size, seq_len)

  return batch


def get_batch_copy(vocab_size, batch_size, seq_len):
  """Generates random data for copying."""
  batch = np.random.choice(
      vocab_size - 1, size=[batch_size, seq_len // 2 - 1]) + 1
  batch = np.concatenate([np.zeros([batch_size, 1], dtype=int), batch], axis=1)
  batch = np.concatenate([batch] * 2, axis=1)

  batch_mask = np.concatenate([
      np.zeros([batch_size, seq_len // 2], dtype=bool),
      np.ones([batch_size, seq_len // 2], dtype=bool)
  ],
                              axis=1)

  return batch, batch_mask


def get_enwik8():
  """Download here: http://prize.hutter1.net/ and put into /data/ folder."""
  with gzip.open('./data/enwik8.gz') as f:
    data = np.fromstring(f.read(int(95e6)), dtype=np.uint8)

  train_data, val_data = np.split(data, [int(90e6)])

  return train_data, val_data


def get_ptb():
  """Download here: https://github.com/wojzaremba/lstm/tree/master/data and put into /data/ folder."""
  with open('./data/ptb.train.txt', 'r') as f:
    train_data = np.fromstring(f.read(), dtype=np.uint8)

  with open('./data/ptb.valid.txt', 'r') as f:
    val_data = np.fromstring(f.read(), dtype=np.uint8)

  return train_data, val_data


def set_default_flags(arg_code):
  """Sets default arguments used in paper."""
  possible_flags = []
  obj_class = collections.namedtuple(
      'obj',
      'dataset seq_len batch_size learning_rate step_size hidden_dim n_layers ffn_dim n_heads feature_type compute_type weight_decay iters_count finetune on_gptln'
  )

  for step_size, finetune in [(-1, False), (512, False), (256, False),
                              (512, True), (256, True)]:
    obj = obj_class(
        dataset='ptb',
        seq_len=1024,
        batch_size=1,
        learning_rate=1e-4,
        step_size=step_size,
        hidden_dim=512,
        n_layers=3,
        ffn_dim=2048,
        n_heads=8,
        feature_type='sqr',
        compute_type='iter',
        weight_decay=0.0,
        iters_count=30000,
        finetune=finetune,
        on_gptln=True)

    possible_flags.append(obj)

  for step_size, finetune in [(-1, False), (2048, False), (1366, False),
                              (2048, True), (1366, True)]:
    obj = obj_class(
        dataset='enwik8',
        seq_len=4096,
        batch_size=1,
        learning_rate=2e-4,
        step_size=step_size,
        hidden_dim=1024,
        n_layers=3,
        ffn_dim=4096,
        n_heads=16,
        feature_type='sqr',
        compute_type='iter',
        weight_decay=0.0,
        iters_count=100000,
        finetune=finetune,
        on_gptln=True)

    possible_flags.append(obj)

  for step_size, finetune in [(-1, False), (128, False), (64, False),
                              (128, True), (64, True)]:
    obj = obj_class(
        dataset='copy',
        seq_len=512,
        batch_size=1,
        learning_rate=1e-2,
        step_size=step_size,
        hidden_dim=256,
        n_layers=1,
        ffn_dim=1024,
        n_heads=4,
        feature_type='sqr',
        compute_type='iter',
        weight_decay=0.0,
        iters_count=15000,
        finetune=finetune,
        on_gptln=False)

    possible_flags.append(obj)

  chosen_flags = possible_flags[arg_code]

  for flag_name, flag_value in chosen_flags._asdict().iteritems():
    setattr(FLAGS, flag_name, flag_value)


def main(_):

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

  if FLAGS.arg_code != -1:
    set_default_flags(FLAGS.arg_code)

  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  if FLAGS.dataset == 'enwik8':
    train_data, val_data = get_enwik8()
  elif FLAGS.dataset == 'ptb':
    train_data, val_data = get_ptb()

  logging.info('Data loaded: %d train chars, %d val chars', len(train_data),
               len(val_data))

  model = slim_performer_model.SLiMPerformer(FLAGS.vocab_size, FLAGS.hidden_dim,
                                             FLAGS.n_layers, FLAGS.ffn_dim,
                                             FLAGS.n_heads, FLAGS.feature_type,
                                             FLAGS.compute_type,
                                             FLAGS.on_gptln).cuda()

  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=FLAGS.learning_rate,
      weight_decay=FLAGS.weight_decay)

  if FLAGS.dataset == 'copy':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10000], gamma=0.1)

  training_start = time.time()

  if FLAGS.dataset == 'copy':
    model.train()

  for train_index in range(FLAGS.iters_count):
    if FLAGS.dataset != 'copy':
      model.train()

    if FLAGS.dataset == 'copy':
      train_batch, mask = get_batch_copy(FLAGS.vocab_size, FLAGS.batch_size,
                                         FLAGS.seq_len)
      mask = torch.from_numpy(mask).cuda()
    else:
      train_batch = get_batch(train_data, FLAGS.batch_size, FLAGS.seq_len,
                              train_index)
    train_batch = torch.from_numpy(train_batch).cuda().long()

    if FLAGS.dataset == 'copy':
      if FLAGS.step_size != -1 and (train_index >= FLAGS.iters_count // 2 or
                                    not FLAGS.finetune):
        train_loss, acc = model.loss_with_grad(
            train_batch, FLAGS.step_size, return_acc=True, nonpad_mask=mask)
      else:
        train_loss, acc = model.full_loss(
            train_batch, with_grad=True, return_acc=True, nonpad_mask=mask)

    else:
      if FLAGS.step_size != -1 and (train_index >= FLAGS.iters_count // 2 or
                                    not FLAGS.finetune):
        train_loss = model.loss_with_grad(train_batch, FLAGS.step_size)
      else:
        train_loss = model.full_loss(train_batch, with_grad=True)

    optimizer.step()
    optimizer.zero_grad()
    if FLAGS.dataset == 'copy':
      scheduler.step()

    train_bpd = train_loss.item() / np.log(2)
    gb_in_use = torch.cuda.max_memory_allocated(0) / (1024 * 1024 * 1024)

    if FLAGS.dataset == 'copy':
      if (train_index + 1) % FLAGS.val_step == 0:

        seconds = time.time() - training_start

        if (train_index + 1) % FLAGS.print_step == 0:
          logging.info(
              'iter#{0} sec={1:.1f} bpd={2:.4f} acc={3:.4f} gb={4:.4f}'.format(
                  train_index + 1, seconds, train_bpd, acc.item(), gb_in_use))

    else:
      if (train_index + 1) % FLAGS.val_step == 0:

        model.eval()

        val_batch = get_batch(val_data, FLAGS.batch_size, FLAGS.seq_len,
                              train_index // FLAGS.val_step)
        val_batch = torch.from_numpy(val_batch).cuda().long()

        with torch.no_grad():
          val_loss = model.full_loss(val_batch, with_grad=False)

        val_bpd = val_loss.item() / np.log(2)

        seconds = time.time() - training_start

        if (train_index + 1) % FLAGS.print_step == 0:
          logging.info(
              'iter#{0} sec={1:.1f} t_bpd={2:.4f} v_bpd={3:.4f} gb={4:.4f}'
              .format(train_index + 1, seconds, train_bpd, val_bpd, gb_in_use))


if __name__ == '__main__':
  app.run(main)
