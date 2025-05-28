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

"""Functions for training and evaluation."""

import copy
import sys
import typing
import numpy as np
import torch
from gradient_coresets_replay.datasets import get_dataset
from gradient_coresets_replay.datasets.utils.continual_dataset import ContinualDataset
from gradient_coresets_replay.utils import loggers
from gradient_coresets_replay.utils.continual_model import ContinualModel

Any = typing.Any
Tuple = typing.Tuple


def mask_classes(
    outputs: torch.Tensor, dataset: ContinualDataset, k: int
) -> None:
  """Given the output tensor, the dataset at hand and the current task, masks the former by setting the responses for the other tasks at -inf.

  It is used to obtain the results for the task-il setting.

  Args:
    outputs: the output tensor
    dataset: the continual dataset
    k: the task index
  """
  outputs[:, 0 : k * dataset.n_classes_per_task] = -float('inf')
  outputs[
      :,
      (k + 1)
      * dataset.n_classes_per_task : dataset.n_tasks
      * dataset.n_classes_per_task,
  ] = -float('inf')


def evaluate(
    model: ContinualModel, dataset: ContinualDataset, last=False
) -> Tuple[list[float], list[float]]:
  """Evaluates the accuracy of the model for each past task.

  Args:
    model: the model to be evaluated
    dataset: the continual dataset at hand
    last: if True, evaluates only on the last dataset in the continual dataset

  Returns:
    a tuple of lists, containing the class-il and task-il accuracy for each task
  """
  status = model.net.training
  model.net.eval()
  accs, accs_mask_classes = [], []
  for k, test_loader in enumerate(dataset.test_loaders):
    if last and k < len(dataset.test_loaders) - 1:
      continue
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    for data in test_loader:
      inputs, labels = data
      inputs, labels = inputs.to(model.device), labels.to(model.device)
      if 'class-il' not in model.compatibility:
        outputs = model(inputs, k)
      else:
        outputs = model(inputs)

      _, pred = torch.max(outputs.data, 1)
      correct += torch.sum(pred == labels).item()
      total += labels.shape[0]

      if dataset.setting == 'class-il':
        mask_classes(outputs, dataset, k)
        _, pred = torch.max(outputs.data, 1)
        correct_mask_classes += torch.sum(pred == labels).item()

    accs.append(
        correct / total * 100 if 'class-il' in model.compatibility else 0
    )
    accs_mask_classes.append(correct_mask_classes / total * 100)

  model.net.train(status)
  return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset, args: Any) -> None:
  """The training process, including evaluations and loggers.

  Args:
    model: the module to be trained
    dataset: the continual dataset at hand
    args: the arguments of the current execution
  """
  model.net.to(model.device)
  temp_model = copy.deepcopy(model.net)
  temp_model_params = copy.deepcopy(temp_model.state_dict())

  results, results_mask_classes = [], []
  results_copy, results_mask_classes_copy = [], []

  model_stash = loggers.create_stash(model, args, dataset)
  csv_logger = None
  if args.csv_log:
    csv_logger = loggers.CsvLogger(dataset.setting, dataset.name, model.name)

  dataset_copy = get_dataset(args)
  for t in range(dataset.n_tasks):
    model.net.train()
    _, _ = dataset_copy.get_data_loaders()
  random_results_class, random_results_task = evaluate(model, dataset_copy)

  print(file=sys.stderr)
  for t in range(dataset.n_tasks):
    model.net.train()
    train_loader, _ = dataset.get_data_loaders()
    if hasattr(model, 'begin_task'):
      model.begin_task(dataset)
    if t:
      accs = evaluate(model, dataset, last=True)
      results[t - 1] = results[t - 1] + accs[0]
      if dataset.setting == 'class-il':
        results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]
    for epoch in range(args.n_epochs):
      for i, data in enumerate(train_loader):
        if hasattr(dataset.train_loader.dataset, 'logits'):
          inputs, labels, not_aug_inputs, _ = data
        else:
          inputs, labels, not_aug_inputs = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        not_aug_inputs = not_aug_inputs.to(model.device)
        loss = model.observe(inputs, labels, not_aug_inputs)

        loggers.progress_bar(i, len(train_loader), epoch, t, loss)

        model_stash['batch_idx'] = i + 1
      model_stash['epoch_idx'] = epoch + 1
      model_stash['batch_idx'] = 0
    model_stash['task_idx'] = t + 1
    model_stash['epoch_idx'] = 0
    if model.name == 'gcr':
      if args.selection_strategy in ['gradmatch']:
        print('using new model')
        temp_model = copy.deepcopy(model.net)
        temp_model_params = copy.deepcopy(temp_model.state_dict())
      model.buffer.add_buffer_data(temp_model, temp_model_params)
    if hasattr(model, 'end_task'):
      model.end_task(dataset)

    accs = evaluate(model, dataset)
    results.append(accs[0])
    results_copy.append(accs[0])
    results_mask_classes.append(accs[1])
    results_mask_classes_copy.append(accs[1])
    mean_acc = np.mean(accs, axis=1)
    loggers.print_mean_accuracy(mean_acc, t + 1, dataset.setting)

    model_stash['mean_accs'].append(mean_acc)
    if args.csv_log:
      csv_logger.log(mean_acc)

  if args.csv_log:
    csv_logger.add_bwt(results, results_mask_classes)
    csv_logger.add_forgetting(results, results_mask_classes)
    csv_logger.add_fwt(
        results,
        random_results_class,
        results_mask_classes,
        random_results_task,
    )
    args.results = str(results_copy)
    args.results_mask_classes = str(results_mask_classes_copy)

  if args.csv_log:
    csv_logger.write(args)


def streaming(
    model: ContinualModel, dataset: ContinualDataset, args: Any
) -> None:
  """The training process, including evaluations and loggers.

  Args:
    model: the module to be trained
    dataset: the continual dataset at hand
    args: the arguments of the current execution
  """
  model.net.to(model.device)
  temp_model = copy.deepcopy(model.net)
  temp_model_params = copy.deepcopy(temp_model.state_dict())

  results, results_mask_classes = [], []
  results_copy, results_mask_classes_copy = [], []

  model_stash = loggers.create_stash(model, args, dataset)
  csv_logger = None
  if args.csv_log:
    csv_logger = loggers.CsvLogger(dataset.setting, dataset.name, model.name)

  print(file=sys.stderr)
  for t in range(dataset.NUM_STREAMS):
    model.net.train()
    train_loader = dataset.get_stream_dataloader()
    if hasattr(model, 'begin_task'):
      model.begin_task(dataset)

    for epoch in range(args.n_epochs):
      for i, data in enumerate(train_loader):
        if hasattr(dataset.train_loader.dataset, 'logits'):
          inputs, labels, not_aug_inputs, _ = data
        else:
          inputs, labels, not_aug_inputs = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        not_aug_inputs = not_aug_inputs.to(model.device)
        loss = model.observe(inputs, labels, not_aug_inputs)

        loggers.progress_bar(i, len(train_loader), epoch, t, loss)

        model_stash['batch_idx'] = i + 1
      model_stash['epoch_idx'] = epoch + 1
      model_stash['batch_idx'] = 0
    model_stash['task_idx'] = t + 1
    model_stash['epoch_idx'] = 0
    if model.name == 'gcr':
      if args.selection_strategy in ['gradmatch']:
        print('using new model')
        temp_model = copy.deepcopy(model.net)
        temp_model_params = copy.deepcopy(temp_model.state_dict())
      model.buffer.add_buffer_data(temp_model, temp_model_params)
    if hasattr(model, 'end_task'):
      model.end_task(dataset)

    accs = evaluate(model, dataset)
    results.append(accs[0])
    results_copy.append(accs[0])
    results_mask_classes.append(accs[1])
    results_mask_classes_copy.append(accs[1])
    mean_acc = np.mean(accs, axis=1)
    loggers.print_mean_accuracy(mean_acc, t + 1, dataset.setting)

    model_stash['mean_accs'].append(mean_acc)
    if args.csv_log:
      csv_logger.log(mean_acc)

  if args.csv_log:
    args.results = str(results_copy)
    args.results_mask_classes = str(results_mask_classes)

  if args.csv_log:
    csv_logger.write(args)
