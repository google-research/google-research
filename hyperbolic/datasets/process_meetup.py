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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collaborative Filtering meetup dataset pre-processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf
from hyperbolic.utils.preprocess import process_dataset
from hyperbolic.utils.preprocess import save_as_pickle


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_path',
    default='data/meetup/',
    help='Path to raw dataset dir')
flags.DEFINE_string(
    'save_dir_path',
    default='data/meetup20_nrand/',
    help='Path to saving directory')


def read_event_times(dataset_path):
  """Maps events times to a dictonary."""
  event_times = {}
  for split in ['train', 'test']:
    path = os.path.join(dataset_path, 'NYC', split, 'events.txt')
    with tf.gfile.Open(path, 'r') as lines:
      for line in lines:
        line = line.strip('\n').split(' ')
        event = line[0]
        timestamp = int(line[2])
        event_times[event] = timestamp
  return event_times


def to_np_new_ids(examples):
  """Creates new ids to a user-events dict. Casts new values as Numpy arrays."""
  user_id = {user: i for i, user in enumerate(examples.keys())}
  all_events = set().union(*examples.values())
  event_id = {event: i for i, event in enumerate(all_events)}
  examples_new_ids = {}
  for user in examples:
    events = [event_id[event] for event in examples[user]]
    examples_new_ids[user_id[user]] = np.array(events)
  return examples_new_ids


def meetup_to_dict(dataset_path, min_interaction=20):
  """Maps raw dataset file to a Dictonary.

  Args:
    dataset_path: Path to directory so that:
      dataset_file/NYC/train/event_users.txt and
      dataset_file/NYC/test/event_users.txt
      both have format of
      event_id user_id user_id ... user_id

      dataset_file/NYC/train/events.txt and
      dataset_file/NYC/test/events.txt
      both have format of
      Event_id Venue_id Time Group_id
      where the format of Time is YYYYMMDDhhmmss.
    min_interaction: number of minimal interactions per user to filter on.

  Returns:
    Dictionary containing users as keys, and a numpy array of events the user
    interacted with, sorted by the time of interaction.
  """
  # create user to event dict
  all_examples = {}
  for split in ['train', 'test']:
    path = os.path.join(dataset_path, 'NYC', split, 'event_users.txt')
    with tf.gfile.Open(path, 'r') as lines:
      for line in lines:
        line = line.strip('\n').split(' ')
        event = line[0]
        for user in line[1:]:
          if user in all_examples:
            all_examples[user].append(event)
          else:
            all_examples[user] = [event]
  # filter on users with enough events and sort events by time
  event_times = read_event_times(dataset_path)
  for user in list(all_examples):
    if len(all_examples[user]) >= min_interaction:
      all_examples[user] = sorted(
          all_examples[user],
          key=lambda event: event_times[event] if event in event_times else 0)
    else:
      del all_examples[user]
  return to_np_new_ids(all_examples)


def main(_):
  dataset_path = FLAGS.dataset_path
  save_path = FLAGS.save_dir_path
  sorted_dict = meetup_to_dict(dataset_path)
  dataset_examples = process_dataset(sorted_dict, random=False)
  save_as_pickle(save_path, dataset_examples)


if __name__ == '__main__':
  app.run(main)
