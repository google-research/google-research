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

"""Gradient Coresets Replay."""

import os

from absl import app
from absl import flags
from six.moves import urllib

from gradient_coresets_replay.datasets import ContinualDataset
from gradient_coresets_replay.datasets import get_dataset
from gradient_coresets_replay.models.gcr import GCR
from gradient_coresets_replay.utils import training
from gradient_coresets_replay.utils.conf import set_random_seed

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

args = flags.FLAGS
# experiment_args
flags.DEFINE_string('output_dir', '/tmp/', 'output dir')
flags.DEFINE_string('model', 'gcr', 'model name')
flags.DEFINE_string(
    'dataset', 'seq-cifar10', 'Which dataset to perform experiments on'
)
flags.DEFINE_float('lr', 0.03, 'lr')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('n_epochs', 1, 'n_epochs')
flags.DEFINE_boolean('imbalanced', False, 'imbalanced dataset')
flags.DEFINE_integer('limit_per_task', 10000, 'number of examples per task')
flags.DEFINE_integer(
    'task_imbalance', 0, 'number of data points for last task for imbalance'
)
flags.DEFINE_boolean('streaming', False, 'data streaming')
flags.DEFINE_integer('stream_batch_size', 0, 'stream_batch_size')

# management_args
flags.DEFINE_integer('seed', 0, 'The random seed')
flags.DEFINE_string('notes', 'None', 'Notes for this run')
flags.DEFINE_boolean('validation', False, 'Test on the validation set')
flags.DEFINE_boolean('csv_log', True, 'Enable csv logging')

# rehearsal_args
flags.DEFINE_integer('buffer_size', 200, 'the size of the memory buffer')
flags.DEFINE_integer(
    'minibatch_size', 32, 'the batch size of the memory buffer'
)

flags.DEFINE_float('alpha', 0.1, 'Penalty weight')
flags.DEFINE_float('beta', 0.5, 'Penalty weight')
flags.DEFINE_float('gamma', 0, 'Penalty weight')
flags.DEFINE_integer('reservoir_size', 200, 'the size of the candidate buffer')
flags.DEFINE_string(
    'selection_strategy',
    'gradmatch',
    'selection strategy for coreset selection',
)

flags.DEFINE_string('results', '', 'Notes for this run')
flags.DEFINE_string('results_mask_classes', '', 'Notes for this run')
flags.DEFINE_integer('serial', '0', 'serial number')


def lecun_fix():
  # Yann moved his website to CloudFlare. You need this now

  opener = urllib.request.build_opener()
  opener.addheaders = [('User-agent', 'Mozilla/5.0')]
  urllib.request.install_opener(opener)


def main(_):
  lecun_fix()

  if args.seed is not None:
    set_random_seed(args.seed)

  dataset = get_dataset(args)
  backbone = dataset.get_backbone()
  loss = dataset.get_loss()
  model = GCR(
      backbone,
      loss,
      dataset.get_transform(),
      dataset.get_barlow_transform(),
      int(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS),
  )
  print(args)
  if args.imbalanced:
    assert args.task_imbalance != 0
  if args.streaming:
    assert args.stream_batch_size != 0
  if isinstance(dataset, ContinualDataset):
    if args.streaming:
      training.streaming(model, dataset, args)
    else:
      training.train(model, dataset, args)


if __name__ == '__main__':
  app.run(main)
