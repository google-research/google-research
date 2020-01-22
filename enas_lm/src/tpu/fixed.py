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

"""Entry point for AWD ENAS with a fixed architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

gfile = tf.gfile
from enas_lm.src.tpu import data_utils
from enas_lm.src.tpu import fixed_lib
from enas_lm.src.tpu import utils


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', '')
flags.DEFINE_string('task_mode', None, 'Must be `train, valid, test`')
flags.DEFINE_string('tpu_job_name', 'train_tpu', '')
flags.DEFINE_integer('save_every', 500, '')

flags.DEFINE_boolean('use_tpu', True, '')
flags.DEFINE_boolean('use_vizier', False, '')

flags.DEFINE_boolean('reset_output_dir', False, '')
flags.DEFINE_string('output_dir', None, '')
flags.DEFINE_string('data_path', None, '')

flags.DEFINE_integer('log_every', 50, '')

MAX_RETRIES = 10


def train_driver(params, model_dir, input_fn, model_fn):
  """What we will do for training."""

  estimator = utils.create_estimator(params, model_dir, model_fn)

  tf.logging.info('Train for {0} steps.'.format(params.num_train_steps))
  for trial_id in range(MAX_RETRIES):
    try:
      estimator.train(input_fn=input_fn, max_steps=params.num_train_steps)
      break
    except Warning as w:
      tf.logging.info(w)
    except Exception as e:  # pylint: disable=broad-except
      tf.logging.info(e)
      traceback.print_exc()
      tf.logging.info('Failed {0} times. Retry!'.format(trial_id+1))
      continue
  else:
    with gfile.GFile(os.path.join(model_dir, 'done'), 'w') as fout:
      fout.write('Job failed after {0} retries'.format(MAX_RETRIES))
      fout.flush()
    sys.exit(1)

  with gfile.GFile(os.path.join(model_dir, 'done'), 'w') as fout:
    fout.write('Job finished')
    fout.flush()


def eval_driver(params, checkpoint_dir, model_dir, input_fn, model_fn):
  """Eval.

  Args:
    params: hyper-parameters.
    checkpoint_dir: where the checkpoints live and where `done` is found.
    model_dir: where to dump eval TensorBoard logs.
    input_fn: for `Estimator`.
    model_fn: for `Estimator`.

  """

  estimator = utils.create_estimator(params, model_dir, model_fn)
  eval_hooks = []


  prev_checkpoint = None
  num_mins_waited = 0
  while True:
    curr_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if curr_checkpoint is not None and curr_checkpoint != prev_checkpoint:
      tf.logging.info('Eval at {0}'.format(curr_checkpoint))
      checkpoint_number = int(curr_checkpoint.split('/')[-1].split('-')[-1])
      if (checkpoint_number >= params.start_moving_average and
          not params.moving_average):
        tf.logging.info('From now on, use moving average for eval')
        del estimator
        params.set_hparam('moving_average', True)
        estimator = utils.create_estimator(params, model_dir, model_fn)
      try:
        results = estimator.evaluate(input_fn=input_fn,
                                     steps=params.num_eval_steps,
                                     hooks=eval_hooks,
                                     checkpoint_path=curr_checkpoint)
        log_ppl = results['log_ppl/{0}'.format(params.task_mode)]
        ppl = np.exp(log_ppl)
        tf.logging.info('Eval step={0} {1}_ppl={2:<.2f}'.format(
            results['global_step'], params.task_mode, ppl))
      except Warning as w:
        tf.logging.info(w)
      except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        tf.logging.info('Eval failed. Retrying...')
        continue
      prev_checkpoint = curr_checkpoint
      num_mins_waited = 0
    elif gfile.Exists(os.path.join(checkpoint_dir, 'done')):
      tf.logging.info('Finished')
      sys.exit(0)
    else:
      time.sleep(30)
      num_mins_waited += 0.5
      tf.logging.info('Waited {0:<.1f} mins'.format(num_mins_waited))
      if num_mins_waited >= 120:
        sys.exit(0)


def main(unused_args):

  output_dir = FLAGS.output_dir

  tf.logging.info('-' * 80)
  if not gfile.IsDirectory(output_dir):
    tf.logging.info('Path {} does not exist. Creating'.format(output_dir))
    gfile.MakeDirs(output_dir)
  elif FLAGS.task_mode == 'train' and FLAGS.reset_output_dir:
    tf.logging.info('Path {} exists. Reseting'.format(output_dir))
    gfile.DeleteRecursively(output_dir)
    gfile.MakeDirs(output_dir)

  params = contrib_training.HParams(
      data_path=FLAGS.data_path,
      log_every=FLAGS.log_every,
      output_dir=output_dir,
      task_mode=FLAGS.task_mode,
      save_every=FLAGS.save_every,
      use_tpu=FLAGS.use_tpu,
      tpu_job_name=FLAGS.tpu_job_name,
      moving_average=False,
      master=FLAGS.master,

  )
  tf.logging.info('fixed_arc={0}'.format(FLAGS.fixed_arc))
  params = fixed_lib.set_default_params(params)
  tf.logging.info('-' * 80)
  tf.logging.info('HParams:\n{0}'.format(params.to_json(indent=2,
                                                        sort_keys=True)))

  input_fn = data_utils.input_fn
  model_fn = fixed_lib.model_fn
  if params.task_mode == 'train':
    train_driver(params,
                 model_dir=output_dir,
                 input_fn=input_fn,
                 model_fn=model_fn)
  else:
    eval_driver(params,
                checkpoint_dir=output_dir,
                model_dir=os.path.join(output_dir, params.task_mode),
                input_fn=input_fn,
                model_fn=model_fn)


if __name__ == '__main__':
  tf.app.run()
