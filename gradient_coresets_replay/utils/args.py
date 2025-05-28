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

"""Argument Parser."""

import argparse
from continual_learning_rishabh.datasets import NAMES as DATASET_NAMES
from continual_learning_rishabh.models import get_all_models


def add_experiment_args(parser):
  """Adds the arguments used by all the models."""
  parser.add_argument(
      '--dataset',
      type=str,
      required=True,
      choices=DATASET_NAMES,
      help='Which dataset to perform experiments on.',
  )
  parser.add_argument(
      '--model',
      type=str,
      required=True,
      help='Model name.',
      choices=get_all_models(),
  )

  parser.add_argument('--lr', type=float, required=True, help='Learning rate.')
  parser.add_argument(
      '--batch_size', type=int, required=True, help='Batch size.'
  )
  parser.add_argument(
      '--n_epochs',
      type=int,
      required=True,
      help='The number of epochs for each task.',
  )
  parser.add_argument(
      '--imbalanced', action='store_true', help='imbalanced data classes'
  )
  parser.add_argument(
      '--limit_per_task',
      type=int,
      required=False,
      default=10000,
      help='number of data points per task',
  )
  parser.add_argument(
      '--task_imbalance',
      type=int,
      default=0,
      help='number of data points for last task for imbalance',
  )
  parser.add_argument('--streaming', action='store_true', help='data streaming')
  parser.add_argument(
      '--stream_batch_size',
      type=int,
      default=0,
      help='number of data points for last task for imbalance',
  )


def add_management_args(parser):
  """Adds the arguments used for logging and saving results."""
  parser.add_argument('--seed', type=int, default=None, help='The random seed.')
  parser.add_argument(
      '--notes', type=str, default=None, help='Notes for this run.'
  )

  parser.add_argument(
      '--csv_log', action='store_true', help='Enable csv logging'
  )
  parser.add_argument(
      '--tensorboard', action='store_true', help='Enable tensorboard logging'
  )
  parser.add_argument(
      '--validation', action='store_true', help='Test on the validation set'
  )


def add_rehearsal_args(parser):
  """Adds the arguments used by all the rehearsal-based methods."""
  parser.add_argument(
      '--buffer_size',
      type=int,
      required=True,
      help='The size of the memory buffer.',
  )
  parser.add_argument(
      '--minibatch_size',
      type=int,
      required=True,
      help='The batch size of the memory buffer.',
  )
