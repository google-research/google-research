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

"""Implementation of Weighted Coresets."""

import math
from typing import Tuple
from continual_learning_rishabh.utils.select_subset import select_subset
import numpy as np
import torch
from torchvision import transforms


def reservoir(num_seen_examples, buffer_size):
  if num_seen_examples < buffer_size:
    return num_seen_examples

  rand = np.random.randint(0, num_seen_examples + 1)
  if rand < buffer_size:
    return rand
  else:
    return -1


class Coreset:
  """The memory buffer of rehearsal method."""

  def __init__(self, num_classes, buffer_size, reservoir_size, device, args):
    self.args = args
    self.task_num = 0
    self.num_classes = num_classes
    self.task_start = []
    self.task_end = []
    self.buffer_size = buffer_size
    self.reservoir_size = reservoir_size
    self.device = device
    self.num_seen_examples = 0
    self.total_seen_examples = 0
    self.buffer_attributes = [
        'buffer_examples',
        'buffer_logits',
        'buffer_labels',
        'buffer_weights',
    ]
    self.reservoir_attributes = ['examples', 'logits', 'labels', 'weights']
    self.transform = None

  def init_tensors(self, examples, labels, logits, weights):
    """Initializes the reservoir and buffer."""
    for attr_str, attr in zip(
        self.reservoir_attributes, [examples, logits, labels, weights]
    ):
      if attr is not None and not hasattr(self, attr_str):
        typ = torch.int64 if attr_str.endswith('els') else torch.float32
        device = self.device
        setattr(
            self,
            attr_str,
            torch.zeros(
                (self.reservoir_size, *attr.shape[1:]), dtype=typ, device=device
            ),
        )
        setattr(
            self,
            'buffer_' + attr_str,
            torch.zeros(
                (self.buffer_size, *attr.shape[1:]), dtype=typ, device=device
            ),
        )

  def add_reservoir_data(
      self, examples, labels=None, logits=None, weights=None
  ):
    """randomly adds examples to the reservoir."""
    if not hasattr(self, 'examples'):
      self.init_tensors(examples, labels, logits, weights)
    if self.args.n_epochs > 1:
      filtered_indices = torch.argmax(logits, dim=1) == labels
      examples, labels, logits, weights = (
          examples[filtered_indices],
          labels[filtered_indices],
          logits[filtered_indices],
          weights[filtered_indices],
      )
    for i in range(examples.shape[0]):
      index = reservoir(self.num_seen_examples, self.examples.shape[0])
      self.num_seen_examples += 1
      if index >= 0:
        self.examples[index] = examples[i].to(self.device)
        if labels is not None:
          self.labels[index] = labels[i].to(self.device)
        if logits is not None:
          self.logits[index] = logits[i].to(self.device)
        if weights is not None:
          self.weights[index] = weights[i].to(self.device)

  def add_buffer_data(self, model, model_params):
    """selects data from the reservoir and adds to the buffer."""
    self.task_num += 1
    index = 0
    self.size_per_task = self.buffer_size // (self.task_num)
    assert self.num_seen_examples >= self.size_per_task
    used_reservoir = min(self.num_seen_examples, self.reservoir_size)
    for i in range(self.task_num - 1):
      chosen_inds, w = select_subset(
          model,
          model_params,
          self.buffer_examples[self.task_start[i] : self.task_end[i]].clone(),
          self.buffer_labels[self.task_start[i] : self.task_end[i]].clone(),
          self.buffer_logits[self.task_start[i] : self.task_end[i]].clone(),
          self.buffer_weights[self.task_start[i] : self.task_end[i]].clone(),
          self.num_classes,
          self.size_per_task,
          self.args.selection_strategy,
          self.transform,
          self.device,
          self.args,
      )
      self.buffer_examples[index : index + self.size_per_task] = (
          self.buffer_examples[self.task_start[i] : self.task_end[i]].clone()[
              chosen_inds
          ]
      )
      self.buffer_labels[index : index + self.size_per_task] = (
          self.buffer_labels[self.task_start[i] : self.task_end[i]].clone()[
              chosen_inds
          ]
      )
      self.buffer_logits[index : index + self.size_per_task] = (
          self.buffer_logits[self.task_start[i] : self.task_end[i]].clone()[
              chosen_inds
          ]
      )
      self.buffer_weights[index : index + self.size_per_task] = torch.tensor(
          w
      ).to(self.device)
      self.task_start[i] = index
      self.task_end[i] = index + self.size_per_task
      index += self.size_per_task
    self.task_start.append(index)
    self.task_end.append(self.buffer_size)
    if self.task_end[-1] - self.task_start[-1] == used_reservoir:
      chosen_inds = np.arange(self.reservoir_size)
      w = torch.ones(len(chosen_inds), device=self.device)
    else:
      chosen_inds, w = select_subset(
          model,
          model_params,
          self.examples.clone()[:used_reservoir],
          self.labels.clone()[:used_reservoir],
          self.logits.clone()[:used_reservoir],
          self.weights.clone()[:used_reservoir],
          self.num_classes,
          self.task_end[-1] - self.task_start[-1],
          self.args.selection_strategy,
          self.transform,
          self.device,
          self.args,
      )

    self.buffer_examples[index:] = self.examples[chosen_inds].to(self.device)
    self.buffer_labels[index:] = self.labels[chosen_inds].to(self.device)
    self.buffer_logits[index:] = self.logits[chosen_inds].to(self.device)
    self.buffer_weights[index:] = torch.tensor(w).to(self.device)
    self.total_seen_examples += self.num_seen_examples
    self.num_seen_examples = 0

  def get_data(
      self, size, transform = None
  ):
    """Returns a batch of data containg a mix of examples from current data, buffer and reservoir."""
    if transform is not None:
      self.transform = transform
    num_reservoir = size
    # factor = 1. - (self.task_num)*0.8/4
    factor = 1 / (self.task_num + 1)
    if num_reservoir > min(
        self.num_seen_examples, math.ceil(self.buffer_size * factor)
    ):
      num_reservoir = min(
          self.num_seen_examples, math.ceil(self.buffer_size * factor)
      )
    prob = 1.0
    if self.task_num == 0:
      num_buffer = 0
    else:
      prob = self.num_seen_examples / (
          self.total_seen_examples + self.num_seen_examples
      )
      num_reservoir = min(num_reservoir, int(size * prob))
      num_buffer = size - num_reservoir
    reservoir_choice = np.random.choice(
        min(self.num_seen_examples, math.ceil(self.buffer_size * factor)),
        size=num_reservoir,
        replace=False,
    )

    if transform is None:
      transform = lambda x: x

    list_examples = [
        transform(ee.cpu()) for ee in self.examples[reservoir_choice]
    ]
    list_logits = self.logits[reservoir_choice]
    list_labels = self.labels[reservoir_choice]
    list_weights = self.weights[reservoir_choice]
    if num_buffer != 0:
      buffer_choice = np.random.choice(
          self.buffer_examples.shape[0], size=num_buffer, replace=False
      )
      list_examples.extend(
          [transform(ee.cpu()) for ee in self.buffer_examples[buffer_choice]]
      )
      list_logits = torch.cat((list_logits, self.buffer_logits[buffer_choice]))
      list_labels = torch.cat((list_labels, self.buffer_labels[buffer_choice]))
      list_weights = torch.cat(
          (list_weights, self.buffer_weights[buffer_choice])
      )
    ret_tuple = (
        torch.stack(list_examples).to(self.device),
        list_labels,
        list_logits,
        list_weights,
    )
    return ret_tuple

  def is_empty(self):
    return self.num_seen_examples == 0 and self.task_num == 0

  def get_all_data(
      self,
  ):
    """Returns full data as tensors."""
    examples = self.examples
    labels = self.labels
    logits = self.logits
    weights = self.weights
    if self.total_seen_examples != 0:
      examples = torch.cat((examples, self.buffer_examples))
      labels = torch.cat((labels, self.buffer_labels))
      logits = torch.cat((logits, self.buffer_logits))
      weights = torch.cat((weights, self.buffer_weights))
    return examples, labels, logits, weights

  def get_barlow_data(self, barlow_transform, size):
    (transform, transform_prime) = barlow_transform

    (examples, l, _, w) = self.get_data(size, None)
    y1 = [transform(ee.cpu()) for ee in examples]
    y2 = [transform_prime(ee.cpu()) for ee in examples]
    return (
        torch.stack(y1).to(self.device),
        torch.stack(y2).to(self.device),
        l,
        w,
    )

  def empty(self):
    for attr_str in self.attributes:
      if hasattr(self, attr_str):
        delattr(self, attr_str)
