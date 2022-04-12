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

"""Compress an existing demo dataset."""

from absl import app
from absl import flags

from rrlfd.bc import pickle_dataset

flags.DEFINE_string('in_path', None, 'Path to dataset to compress.')
flags.DEFINE_string('out_path', None, 'Path to which to write compressed data.')
FLAGS = flags.FLAGS


def compress_dataset(demos_file, new_demos_file):
  dataset = pickle_dataset.DemoReader(path=demos_file)
  writer = pickle_dataset.DemoWriter(path=new_demos_file)

  for obs, act in zip(dataset.observations, dataset.actions):
    writer.write_episode(obs, act)


def main(_):
  compress_dataset(FLAGS.in_path, FLAGS.out_path)


if __name__ == '__main__':
  app.run(main)
