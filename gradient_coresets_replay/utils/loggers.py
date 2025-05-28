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

"""Logging and saving utilites."""

import csv
import datetime
import os
import sys
import typing
import numpy as np
import torch
from gradient_coresets_replay.datasets import ContinualDataset
from gradient_coresets_replay.utils.conf import base_path

nn = torch.nn
Any = typing.Any
Dict = typing.Dict

useless_args = [
    'dataset',
    'tensorboard',
    'validation',
    'model',
    'csv_log',
    'notes',
    'load_best_args',
]


def create_if_not_exists(path: str) -> None:
  """Creates the specified folder if it does not exist.

  Args:
    path: the complete path of the folder to be created
  """
  if not os.path.exists(path):
    os.makedirs(path)


def create_stash(
    model: nn.Module, args: Any, dataset: ContinualDataset
) -> Dict[Any, str]:
  """Creates the dictionary where to save the model status.

  Args:
    model: the model
    args: the current arguments
    dataset: the dataset at hand

  Returns:
    returns dictionary containing model training and performance history.
  """
  now = datetime.datetime.now()
  model_stash = {'task_idx': 0, 'epoch_idx': 0, 'batch_idx': 0}
  name_parts = [args.dataset, model.NAME]
  if 'buffer_size' in vars(args).keys():
    name_parts.append('buf_' + str(args.buffer_size))
  name_parts.append(now.strftime('%Y%m%d_%H%M%S_%f'))
  model_stash['model_name'] = '/'.join(name_parts)
  model_stash['mean_accs'] = []
  model_stash['args'] = args
  model_stash['backup_folder'] = os.path.join(
      base_path(), 'backups', dataset.SETTING, model_stash['model_name']
  )
  return model_stash


def progress_bar(
    i: int,
    max_iter: int,
    epoch: typing.Union[int, str],
    task_number: int,
    loss: float,
) -> None:
  """Prints out the progress bar on the stderr file.

  Args:
    i: the current iteration
    max_iter: the maximum number of iteration
    epoch: the epoch
    task_number: the task index
    loss: the current value of the loss function
  """
  if not (i + 1) % 10 or (i + 1) == max_iter:
    progress = min(float((i + 1) / max_iter), 1)
    progress_b = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
    print(
        '\r[ {} ] Task {} | epoch {}: |{}| loss: {}'.format(
            datetime.datetime.now().strftime('%m-%d | %H:%M'),
            task_number + 1 if isinstance(task_number, int) else task_number,
            epoch,
            progress_b,
            round(loss, 8),
        ),
        file=sys.stderr,
        end='',
        flush=True,
    )


def backward_transfer(results):
  n_tasks = len(results)
  li = list()
  for i in range(n_tasks - 1):
    li.append(results[-1][i] - results[i][i])

  return np.mean(li)


def forward_transfer(results, random_results):
  n_tasks = len(results)
  li = list()
  for i in range(1, n_tasks):
    li.append(results[i - 1][i] - random_results[i])

  return np.mean(li)


def forgetting(results):
  n_tasks = len(results)
  li = list()
  for i in range(n_tasks - 1):
    results[i] += [0.0] * (n_tasks - len(results[i]))
  np_res = np.array(results)
  maxx = np.max(np_res, axis=0)
  for i in range(n_tasks - 1):
    li.append(maxx[i] - results[-1][i])

  return np.mean(li)


def print_mean_accuracy(
    mean_acc: np.ndarray, task_number: int, setting: str
) -> None:
  """Prints the mean accuracy on stderr.

  Args:
    mean_acc: mean accuracy value
    task_number: task index
    setting: the setting of the benchmark
  """
  if setting == 'domain-il':
    mean_acc, _ = mean_acc
    print(
        '\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)
        ),
        file=sys.stderr,
    )
  else:
    mean_acc_class_il, mean_acc_task_il = mean_acc
    print(
        '\nAccuracy for {} task(s): \t [Class-IL]: {} %'
        ' \t [Task-IL]: {} %\n'.format(
            task_number, round(mean_acc_class_il, 2), round(mean_acc_task_il, 2)
        ),
        file=sys.stderr,
    )


class CsvLogger:
  """CSV Logger."""

  def __init__(
      self, setting_str: str, dataset_str: str, model_str: str
  ) -> None:
    self.accs = []
    if setting_str == 'class-il':
      self.accs_mask_classes = []
    self.setting = setting_str
    self.dataset = dataset_str
    self.model = model_str
    self.fwt = None
    self.fwt_mask_classes = None
    self.bwt = None
    self.bwt_mask_classes = None
    self.forgetting = None
    self.forgetting_mask_classes = None
    self.f = 0

  def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
    self.fwt = forward_transfer(results, accs)
    if self.setting == 'class-il':
      self.fwt_mask_classes = forward_transfer(
          results_mask_classes, accs_mask_classes
      )

  def add_bwt(self, results, results_mask_classes):
    self.bwt = backward_transfer(results)
    self.bwt_mask_classes = backward_transfer(results_mask_classes)

  def add_forgetting(self, results, results_mask_classes):
    self.forgetting = forgetting(results)
    self.forgetting_mask_classes = forgetting(results_mask_classes)

  def log(self, mean_acc: np.ndarray) -> None:
    """Logs a mean accuracy value.

    Args:
      mean_acc: mean accuracy value
    """
    if self.setting == 'general-continual':
      self.accs.append(mean_acc)
    elif self.setting == 'domain-il':
      mean_acc, _ = mean_acc
      self.accs.append(mean_acc)
    else:
      mean_acc_class_il, mean_acc_task_il = mean_acc
      self.accs.append(mean_acc_class_il)
      self.accs_mask_classes.append(mean_acc_task_il)

  def write(self, flags: Any) -> None:
    """writes out the logged value along with its arguments.

    Args:
      flags: the arguments of the current experiment
    """

    args = {}

    args['alpha'] = flags.alpha
    args['beta'] = flags.beta
    args['gamma'] = flags.gamma
    args['seed'] = flags.seed
    args['buffer_size'] = flags.buffer_size
    args['reservoir_size'] = flags.reservoir_size
    args['selection_strategy'] = flags.selection_strategy
    args['lr'] = flags.lr
    args['n_epochs'] = flags.n_epochs
    args['imbalanced'] = flags.imbalanced
    args['limit_per_task'] = flags.limit_per_task
    args['task_imbalance'] = flags.task_imbalance
    args['streaming'] = flags.streaming
    args['stream_batch_size'] = flags.stream_batch_size
    args['validation'] = flags.validation
    args['results'] = flags.results
    args['results_mask_classes'] = flags.results_mask_classes
    args['notes'] = flags.notes
    columns = list(args.keys())

    output_dir = '/tmp/' + flags.output_dir
    new_cols = []
    for i, acc in enumerate(self.accs):
      args['task' + str(i + 1)] = acc
      new_cols.append('task' + str(i + 1))

    args['forward_transfer'] = self.fwt
    new_cols.append('forward_transfer')
    args['backward_transfer'] = self.bwt
    new_cols.append('backward_transfer')
    args['forgetting'] = self.forgetting
    new_cols.append('forgetting')

    columns = new_cols + columns

    create_if_not_exists(output_dir + 'results/' + self.setting)
    create_if_not_exists(
        output_dir + 'results/' + self.setting + '/' + self.dataset
    )
    create_if_not_exists(
        output_dir
        + 'results/'
        + self.setting
        + '/'
        + self.dataset
        + '/'
        + self.model
    )

    write_headers = False

    path = (
        output_dir
        + 'results/'
        + self.setting
        + '/'
        + self.dataset
        + '/'
        + self.model
        + f'/mean_accs{flags.serial}.csv'
    )
    if not os.path.exists(path):
      write_headers = True
    with open(path, 'a') as tmp:
      writer = csv.DictWriter(tmp, fieldnames=columns)
      if write_headers:
        writer.writeheader()
      writer.writerow(args)

    if self.setting == 'class-il':
      create_if_not_exists(output_dir + 'results/task-il/' + self.dataset)
      create_if_not_exists(
          output_dir + 'results/task-il/' + self.dataset + '/' + self.model
      )

      for i, acc in enumerate(self.accs_mask_classes):
        args['task' + str(i + 1)] = acc

      write_headers = False

      path = (
          output_dir
          + 'results/task-il'
          + '/'
          + self.dataset
          + '/'
          + self.model
          + f'/mean_accs{flags.serial}.csv'
      )
      if not os.path.exists(path):
        write_headers = True
      with open(path, 'a') as tmp:
        writer = csv.DictWriter(tmp, fieldnames=columns)
        if write_headers:
          writer.writeheader()
        writer.writerow(args)
