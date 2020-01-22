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

r"""A short script for training MAML.

Example to run
python -m norml.train_maml --config MOVE_POINT_ROTATE_MAML
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from dotmap import DotMap
import tensorflow.compat.v1 as tf

from norml import config_maml
from norml import maml_rl


FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'RL_PENDULUM_GYM_CONFIG_META',
                    'Configuration for training.')


def main(argv):
  del argv  # Unused
  config = DotMap(getattr(config_maml, FLAGS.config))
  print('MAML config: %s' % FLAGS.config)
  tf.logging.info('MAML config: %s', FLAGS.config)
  algo = maml_rl.MAMLReinforcementLearning(config)
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True

  with tf.Session(config=sess_config) as sess:
    algo.init_logging(sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    done = False
    while not done:
      done, _ = algo.train(sess, 10)
    algo.stop_logging()


if __name__ == '__main__':
  tf.app.run()
