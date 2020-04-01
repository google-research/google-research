# coding=utf-8
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
"""Main function for the project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

from ieg import options
from ieg.dataset_utils import datasets
from ieg.models.basemodel import BaseModel
from ieg.models.l2rmodel import L2R
from ieg.models.model import IEG

import tensorflow.compat.v1 as tf

logger = tf.get_logger()
logger.propagate = False

FLAGS = flags.FLAGS

options.define_basic_flags()


def train(model, sess):
  """Training launch function."""
  with sess.as_default():
    model.train()


def evaluation(model, sess):
  """Evaluation launch function."""
  with sess.as_default():
    model.evaluation()


def main(_):

  FLAGS.checkpoint_path = os.path.join(
      FLAGS.checkpoint_path,
      '{}_p{}'.format(FLAGS.dataset, FLAGS.probe_dataset_hold_ratio),
      FLAGS.network_name)
  strategy = tf.distribute.MirroredStrategy()
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)

  # Creates dataset
  dataset = datasets.CIFAR()

  if FLAGS.method == 'supervised':
    model = BaseModel(sess=sess, strategy=strategy, dataset=dataset)
  elif FLAGS.method == 'l2r':
    model = L2R(sess=sess, strategy=strategy, dataset=dataset)
  elif FLAGS.method == 'ieg':
    model = IEG(sess=sess, strategy=strategy, dataset=dataset)
  else:
    raise NotImplementedError('{} is not existed'.format(FLAGS.method))

  if FLAGS.mode == 'evaluation':
    evaluation(model, sess)
  else:
    train(model, sess)

if __name__ == '__main__':
  app.run(main)
