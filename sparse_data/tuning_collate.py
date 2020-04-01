# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Collates results from workers for parameter tuning.

Results must have already been created with tuning_worker.py.
"""

import time
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from sparse_data import utils

FILE_PATH = 'trees/out'
SEED = 1232323

FLAGS = flags.FLAGS

flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite existing '
                     'parameters.')
flags.DEFINE_boolean(
    'strict', False, 'Whether to ensure that worker output is '
    'strictly correct; e.g., no outputs are missing.')


def main(_):
  experiment_string = utils.get_experiment_string(
      FLAGS.dataset,
      vec=FLAGS.vec,
      dim_red=FLAGS.dim_red,
      features=FLAGS.features,
      problem=FLAGS.problem,
      alternate=FLAGS.alternate)

  worker_dir = '{}/params/{}/worker_out'.format(FILE_PATH, experiment_string)
  param_dir = '{}/params/{}'.format(FILE_PATH, experiment_string)

  logging.info('\n' * 3)
  logging.info('Collation (parameter search results) for %s', experiment_string)
  logging.info('\n' * 3)

  # worker out path format is {0}/dataset_idx={1}_{2}/{3}/search_idx={4}_{5}.out
  # where 0 = worker_out_dir, 1 = dataset_idx, 2 = num_dataset, 3 = method,
  # 4 = search_idx, 5 = num_search

  # gets idx from a str with format {text}={idx}_{text}
  extract_idx_func = lambda x: int(x.split('=')[1].split('_')[0])

  dataset_ids = sorted(tf.gfile.ListDir(worker_dir), key=extract_idx_func)
  assert dataset_ids  # check non-empty

  num_dataset = int(dataset_ids[0].split('_')[-1])
  if FLAGS.strict:
    assert num_dataset == FLAGS.num_dataset

  for d_idx, dataset_id in enumerate(dataset_ids):
    assert dataset_id == 'dataset_idx={}_{}'.format(d_idx, num_dataset)
    dataset_id_path = '{}/{}'.format(worker_dir, dataset_id)
    methods = tf.gfile.ListDir(dataset_id_path)

    for method in methods:
      # setup directories
      method_path = '{}/{}'.format(dataset_id_path, method)
      save_dir = '{}/{}'.format(param_dir, dataset_id)
      tf.gfile.MakeDirs(save_dir)

      # skip collation if already performed
      save_path = '{}/{}.param'.format(save_dir, method)
      if tf.gfile.Exists(save_path) and not FLAGS.overwrite:
        logging.info('Collation already completed for %s, dataset %d/%d',
                     method, d_idx, num_dataset)
        continue

      search_ids = sorted(tf.gfile.ListDir(method_path), key=extract_idx_func)
      num_search = int(search_ids[0].split('_')[-1].rstrip('.out'))

      start = time.time()

      # look for best and worst tuning scores and save related parameters
      best_score, worst_score = float('-inf'), float('inf')
      best_param_str = None
      best_path = None
      for s_idx, search_id in enumerate(search_ids):
        if FLAGS.strict:
          assert search_id == 'search_idx={}_{}.out'.format(s_idx, num_search)

        read_path = '{}/{}'.format(method_path, search_id)
        with tf.gfile.GFile(read_path, 'r') as f:
          lines = f.read().splitlines()

        score = float(lines[1])
        # assume that scores should be maximized (e.g., negative MSE or acc)
        if score > best_score:
          best_param_str, best_path, best_score = lines[0], read_path, score
        if score < worst_score:  # save worst config for debugging
          worst_score = lines[0], score

      # note: reports number of worker results read which is less than the value
      # of num_search which describes the *maximum* number of worker searches
      logging.info('Collation for method=%s, dataset %d/%d (%d reads, %.3f s)',
                   method, d_idx, num_dataset, len(search_ids),
                   time.time() - start)
      logging.info('best score=%.3f, %s', best_score, best_param_str)
      logging.info('\n' * 2)

      # save best parameters
      tf.gfile.Copy(best_path, save_path, overwrite=True)
      logging.info('Saved collation results to %s', save_path)


if __name__ == '__main__':
  app.run()
