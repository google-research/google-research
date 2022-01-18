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

"""Generate NQ gold cache data."""

import os
import pickle

from absl import app
from absl import flags
from absl import logging
from natural_questions import eval_utils


flags.DEFINE_string(
    'gold_path', None, 'Path to the gzip JSON data. For '
    'multiple files, should be a glob '
    'pattern (e.g. "/path/to/files-*"')

flags.DEFINE_integer('num_threads', 10, 'Number of threads for reading.')

FLAGS = flags.FLAGS


def main(_):
  cache_path = os.path.join(os.path.dirname(FLAGS.gold_path), 'cache')
  nq_gold_dict = eval_utils.read_annotation(
      FLAGS.gold_path, n_threads=FLAGS.num_threads)
  logging.info('Caching gold data to: %s', format(cache_path))
  pickle.dump(nq_gold_dict, open(cache_path, 'wb'))

if __name__ == '__main__':
  flags.mark_flag_as_required('gold_path')
  app.run(main)
