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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf

from axial import config_imagenet32
from axial import config_imagenet64
from axial import datasets
from axial import models
from axial import worker_util

FLAGS = flags.FLAGS
flags.DEFINE_string('master', None, '')
flags.DEFINE_string('logdir', None, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_string('config', None, 'imagenet32 or imagenet64')

flags.mark_flag_as_required('logdir')
flags.mark_flag_as_required('config')


def main(_):
  if FLAGS.config == 'imagenet32':
    config = config_imagenet32.get_config()
  elif FLAGS.config == 'imagenet64':
    config = config_imagenet64.get_config()
  else:
    raise ValueError(config)
  logging.info('config: {}'.format(config))

  # Seeding
  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  # Model
  def model_constructor():
    return getattr(models, config.model_name)(config.model_config)

  # Dataset
  dataset = datasets.get_dataset(config.dataset_name,
                                 **config.dataset_config.values())

  worker_util.run_eval(
      model_constructor=model_constructor,
      logdir=FLAGS.logdir,
      total_bs=config.eval_total_bs,
      master=FLAGS.master,
      input_fn=dataset.eval_input_fn,
      dataset_size=dataset.get_size(is_train=False))


if __name__ == '__main__':
  app.run(main)
