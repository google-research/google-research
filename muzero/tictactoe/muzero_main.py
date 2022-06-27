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

# pylint: disable=unused-argument
# pylint: disable=missing-docstring
"""MuZero."""

from absl import app
from absl import flags
from seed_rl.common import common_flags  # pylint: disable=unused-import
import tensorflow as tf

from muzero import actor
from muzero import core as mzcore
from muzero import learner
from muzero import learner_flags
from muzero.tictactoe import env
from muzero.tictactoe import network


flags.DEFINE_string('optimizer', 'adam', 'One of [sgd, adam, rmsprop, adagrad]')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('momentum', 0.9, 'Momentum')
flags.DEFINE_float('lr_decay_fraction', 0.01,
                   'Final LR as a fraction of initial.')
flags.DEFINE_integer('lr_warm_restarts', 1, 'Do warm restarts for LR decay.')
flags.DEFINE_integer('lr_decay_steps', int(2e4),
                     'Decay steps for the cosine learning rate schedule.')
flags.DEFINE_integer('n_mlp_layers', 2, 'Number of MLP hidden layers.')
flags.DEFINE_integer('mlp_size', 512, 'Sizes of each of MLP hidden layer.')
flags.DEFINE_integer(
    'n_lstm_layers', 2,
    'Number of LSTM layers. LSTM layers afre applied after MLP layers.')
flags.DEFINE_integer('lstm_size', 128, 'Sizes of each LSTM layer.')
flags.DEFINE_integer('n_head_hidden_layers', 2,
                     'Number of hidden layers in heads.')
flags.DEFINE_integer('head_hidden_size', 512,
                     'Sizes of each head hidden layer.')

flags.DEFINE_integer('num_simulations', 64, 'Number of simulations.')
flags.DEFINE_integer('td_steps', -1, 'Number of TD steps.')
flags.DEFINE_integer('num_unroll_steps', 5, 'Number of unroll steps.')
flags.DEFINE_float('one_minus_discount', .003, 'One minus discount factor.')
flags.DEFINE_float('dirichlet_alpha', .5, 'Dirichlet alpha.')
flags.DEFINE_float('root_exploration_fraction', .25,
                   'Root exploration fraction.')
flags.DEFINE_integer('pb_c_base', 19652, 'PB C Base.')
flags.DEFINE_float('pb_c_init', 2.5, 'PB C Init.')

flags.DEFINE_float('temperature', .1, 'for softmax sampling of actions')

flags.DEFINE_integer('value_encoder_steps', 8, 'If 0, take 1 step per integer')
flags.DEFINE_integer('reward_encoder_steps', None,
                     'If None, take over the value from value_encoder_steps')
flags.DEFINE_integer(
    'play_max_after_moves', -1,
    'Play the argmax after this many game moves. -1 means never play argmax')
flags.DEFINE_integer(
    'use_softmax_for_action_selection', 0,
    'Whether to use softmax (1) for regular histogram sampling (0).')

FLAGS = flags.FLAGS


def create_agent(env_descriptor, parametric_action_distribution):
  reward_encoder_steps = FLAGS.reward_encoder_steps
  if reward_encoder_steps is None:
    reward_encoder_steps = FLAGS.value_encoder_steps

  reward_encoder = mzcore.ValueEncoder(
      *env_descriptor.reward_range,
      reward_encoder_steps,
      use_contractive_mapping=False)
  value_encoder = mzcore.ValueEncoder(
      *env_descriptor.value_range,
      FLAGS.value_encoder_steps,
      use_contractive_mapping=False)
  return network.MLPandLSTM(
      mlp_sizes=[FLAGS.mlp_size] * FLAGS.n_mlp_layers,
      parametric_action_distribution=parametric_action_distribution,
      rnn_sizes=[FLAGS.lstm_size] * FLAGS.n_lstm_layers,
      head_hidden_sizes=[FLAGS.head_hidden_size] * FLAGS.n_head_hidden_layers,
      reward_encoder=reward_encoder,
      value_encoder=value_encoder)


def create_optimizer(unused_final_iteration):
  if FLAGS.lr_warm_restarts:
    learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(
        FLAGS.learning_rate,
        FLAGS.lr_decay_steps,
        alpha=FLAGS.lr_decay_fraction)
  else:
    learning_rate_fn = tf.keras.experimental.CosineDecay(
        FLAGS.learning_rate,
        FLAGS.lr_decay_steps,
        alpha=FLAGS.lr_decay_fraction)
  if FLAGS.optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate_fn, momentum=FLAGS.momentum)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.keras.optimizers.AdaGrad(learning_rate_fn)
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate_fn, momentum=FLAGS.momentum)
  else:
    raise ValueError('Unknown optimizer: {}'.format(FLAGS.optimizer))
  return optimizer, learning_rate_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  def visit_softmax_temperature(num_moves, training_steps, is_training=True):
    if not is_training:
      return 0.
    if FLAGS.play_max_after_moves < 0:
      return FLAGS.temperature
    if num_moves < FLAGS.play_max_after_moves:
      return FLAGS.temperature
    else:
      return 0.

  env_descriptor = env.get_descriptor()

  # Known bounds for Q-values have to include rewards and values.
  known_bounds = mzcore.KnownBounds(
      *map(sum, zip(env_descriptor.reward_range, env_descriptor.value_range)))
  mzconfig = mzcore.MuZeroConfig(
      action_space_size=env_descriptor.action_space.n,
      max_moves=env_descriptor.action_space.n,
      discount=1.0 - FLAGS.one_minus_discount,
      dirichlet_alpha=FLAGS.dirichlet_alpha,
      root_exploration_fraction=FLAGS.root_exploration_fraction,
      num_simulations=FLAGS.num_simulations,
      initial_inference_batch_size=(
          learner_flags.INITIAL_INFERENCE_BATCH_SIZE.value),
      recurrent_inference_batch_size=(
          learner_flags.RECURRENT_INFERENCE_BATCH_SIZE.value),
      train_batch_size=learner_flags.BATCH_SIZE.value,
      td_steps=FLAGS.td_steps,
      num_unroll_steps=FLAGS.num_unroll_steps,
      pb_c_base=FLAGS.pb_c_base,
      pb_c_init=FLAGS.pb_c_init,
      known_bounds=known_bounds,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      use_softmax_for_action_selection=(
          FLAGS.use_softmax_for_action_selection == 1),
      max_num_action_expansion=FLAGS.max_num_action_expansion)

  if FLAGS.run_mode == 'actor':
    actor.actor_loop(env.create_environment, mzconfig)
  elif FLAGS.run_mode == 'learner':
    learner.learner_loop(env_descriptor, create_agent, create_optimizer,
                         learner_flags.learner_config_from_flags(), mzconfig)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
