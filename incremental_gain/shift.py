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

# pylint: disable=invalid-name
"""shift.py.

Main entry point for collecting experiment data.
"""

import functools
import pickle
import time

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
# from jax.config import config as jax_config
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from incremental_gain import jax_utils
from incremental_gain import problem_instance_utils
from incremental_gain import training_utils

# jax_config.update('jax_enable_x64', True)


flags.DEFINE_float('learning_rate', 1e-2, 'learning_rate.')
flags.DEFINE_integer('n_shift_epochs', 12, 'n_shift_epochs.')
flags.DEFINE_integer('n_train_epochs', 200, 'n_train_epochs.')
flags.DEFINE_float('alpha', 0.3, 'alpha.')
flags.DEFINE_integer('n_trajs_per_epoch', 10, 'n_trajs_per_epoch.')
flags.DEFINE_integer('n_trajs_final_eval', 500, 'n_trajs_final_eval.')
flags.DEFINE_integer('batch_size', 512, 'batch_size.')
flags.DEFINE_string('config_outfile', None, 'config_outfile.')
flags.DEFINE_string('params_outfile', None, 'params_outfile.')
flags.DEFINE_string('metrics_outfile', None, 'metrics_outfile.')
flags.DEFINE_integer('seed', None, 'seed.')

flags.DEFINE_float('p', 1, 'p.')
flags.DEFINE_integer('state_dim', 10, 'state_dim.')
flags.DEFINE_integer('horizon', 100, 'horizon.')
flags.DEFINE_boolean('aggregate_data', False, 'aggregate_data.')
flags.DEFINE_boolean('dagger', False, 'dagger.')
flags.DEFINE_float('igs_constraint_lam', 0.0, 'igs_constraint_lam.')

flags.DEFINE_boolean('verbose_learner', False, 'verbose_learner.')

FLAGS = flags.FLAGS


def validate_flags():
  """Validate flags."""
  assert FLAGS.learning_rate >= 0.0
  assert FLAGS.n_shift_epochs >= 0
  assert FLAGS.n_train_epochs >= 0
  assert FLAGS.alpha >= 0.0 and FLAGS.alpha <= 1.0
  assert FLAGS.n_trajs_per_epoch >= 0
  assert FLAGS.n_trajs_final_eval >= 0
  assert FLAGS.batch_size >= 0
  assert FLAGS.p > 0.0
  assert FLAGS.state_dim >= 1
  assert FLAGS.horizon >= 1
  assert FLAGS.aggregate_data or not FLAGS.dagger
  assert FLAGS.igs_constraint_lam >= 0.0


def get_config():
  """Get config."""
  keys = (
      'learning_rate',
      'n_shift_epochs',
      'n_train_epochs',
      'alpha',
      'n_trajs_per_epoch',
      'n_trajs_final_eval',
      'batch_size',
      'config_outfile',
      'params_outfile',
      'metrics_outfile',
      'seed',
      'p',
      'state_dim',
      'horizon',
      'aggregate_data',
      'dagger',
      'igs_constraint_lam',
  )
  return {k: getattr(FLAGS, k) for k in keys}


def stats(x):
  return (np.percentile(x, 10), np.median(x), np.percentile(x, 90))


def main(unused_argv):
  validate_flags()
  logging.info(get_config())

  seed = FLAGS.seed if FLAGS.seed is not None else np.random.randint(
      0, 0x7fffffff)
  rng = np.random.RandomState(seed)
  key_seq = hk.PRNGSequence(rng.randint(0, 0x7fffffff))

  activation = jnp.tanh

  eta = 0.3
  assert eta < 4/(5 + FLAGS.p)  # needed for IGS stability

  if FLAGS.dagger:
    intermediate_policy = training_utils.dagger_policy_with_expert
    final_policy = training_utils.dagger_final_policy
  else:
    intermediate_policy = training_utils.mixed_policy_with_expert
    final_policy = training_utils.final_policy

  # make dynamics and expert
  dynamics, expert_policy = problem_instance_utils.make_dynamics_and_expert(
      next(key_seq), FLAGS.state_dim, FLAGS.p, eta, activation)

  policy_net = training_utils.make_policy_net(64, FLAGS.state_dim, activation)

  opt_init, opt_update = optax.chain(
      # Set the parameters of Adam. Note the learning_rate is not here.
      optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
      # Put a minus sign to *minimise* the loss.
      optax.scale(-FLAGS.learning_rate))

  aggregate_states, aggregate_actions = [], []
  policy_params = []  # accumulate all the networks trained at each epoch
  for shift_epoch in tqdm.trange(FLAGS.n_shift_epochs):
    logging.info('starting epoch %d', shift_epoch)

    start_time = time.time()
    if shift_epoch == 0:
      epoch_rollout_policy = expert_policy
    else:
      epoch_rollout_policy = functools.partial(
          intermediate_policy, policy_net, expert_policy,
          policy_params, FLAGS.alpha)

    x0s_epoch = problem_instance_utils.sample_initial_conditions(
        next(key_seq), FLAGS.n_trajs_per_epoch, FLAGS.state_dim)
    Xs_epoch, Us_epoch = jax.vmap(
        problem_instance_utils.rollout_policy,
        in_axes=(None, None, 0, None))(dynamics, epoch_rollout_policy,
                                       x0s_epoch, FLAGS.horizon)
    Us_expert_labels = jax.vmap(
        lambda traj: jax.vmap(expert_policy)(traj[:-1]))(Xs_epoch)

    logging.info('rolling out %d trajectories took %f seconds',
                 FLAGS.n_trajs_per_epoch,
                 time.time() - start_time)

    # compute goal error
    logging.info('goal error: %s',
                 stats(np.linalg.norm(Xs_epoch[:, -1, :], axis=1)))
    # compute imitation error
    logging.info(
        'imitiation error: %s',
        stats(
            np.sum(np.linalg.norm(Us_epoch - Us_expert_labels, axis=2),
                   axis=1)))

    # format for training
    epoch_train_states = Xs_epoch[:, :-1, :].reshape((-1, Xs_epoch.shape[-1]))
    epoch_train_actions = Us_expert_labels.reshape(
        (-1, Us_expert_labels.shape[-1]))

    # aggregate the accumulated data
    if FLAGS.aggregate_data:
      aggregate_states.append(epoch_train_states)
      aggregate_actions.append(epoch_train_actions)
      epoch_train_states = np.concatenate(aggregate_states, axis=0)
      epoch_train_actions = np.concatenate(aggregate_actions, axis=0)

    logging.info('epoch_train_states.shape: %s', epoch_train_states.shape)
    logging.info('epoch_train_actions.shape: %s', epoch_train_actions.shape)
    assert epoch_train_states.shape[0] == epoch_train_actions.shape[0]
    assert epoch_train_states.shape[1] == FLAGS.state_dim
    assert epoch_train_actions.shape[1] == FLAGS.state_dim

    # initial parameters for training
    if shift_epoch == 0:
      params = policy_net.init(next(key_seq), epoch_train_states[0])
      trust_region_params = jax_utils.pytree_zeros_like(params)
    else:
      assert len(policy_params) >= 1
      params = policy_params[-1]
      trust_region_params = params

    if FLAGS.igs_constraint_lam > 0.0:
      if shift_epoch == FLAGS.n_shift_epochs - 1:
        def policy_fn(policy_network, this_policy_params, x):
          return final_policy(policy_network,
                              policy_params + [this_policy_params],
                              FLAGS.alpha,
                              x)
      else:
        def policy_fn(policy_network, this_policy_params, x):
          return intermediate_policy(policy_network,
                                     expert_policy,
                                     policy_params + [this_policy_params],
                                     FLAGS.alpha,
                                     x)
      def igs_loss(x, y, fx, fy):
        # want |fx - fy| - |x - y| <= 0
        ineq = jnp.abs(fx - fy) - jnp.abs(x - y)
        return FLAGS.igs_constraint_lam * jnp.maximum(ineq, 0)
      igs_constraint_args = (dynamics, igs_loss, policy_fn)
    else:
      igs_constraint_args = None

    start_time = time.time()
    params, _, last_epoch_losses = training_utils.train_policy_network(
        policy_net, opt_update, epoch_train_states, epoch_train_actions, params,
        opt_init(params), trust_region_params, 0.0, igs_constraint_args,
        FLAGS.n_train_epochs, FLAGS.batch_size, 0.0, 1000,
        rng, FLAGS.verbose_learner)
    policy_params.append(params)
    logging.info('shift_epoch=%d, last_epoch_losses=%s, '
                 'avg_last_epoch_losses=%s',
                 shift_epoch,
                 last_epoch_losses,
                 last_epoch_losses / len(epoch_train_states))
    logging.info('train_policy_network at epoch %d took %f seconds',
                 shift_epoch,
                 time.time() - start_time)

  logging.info('running final episodes')

  x0s_final_test = problem_instance_utils.sample_initial_conditions(
      next(key_seq), FLAGS.n_trajs_final_eval, FLAGS.state_dim)
  Xs_final_test_shift, Us_final_test_shift = jax.vmap(
      problem_instance_utils.rollout_policy,
      in_axes=(None, None, 0,
               None))(dynamics,
                      functools.partial(final_policy, policy_net,
                                        policy_params, FLAGS.alpha),
                      x0s_final_test, FLAGS.horizon)
  Us_expert_final_test_shift = jax.vmap(
      lambda traj: jax.vmap(expert_policy)(traj[:-1]))(Xs_final_test_shift)

  Xs_final_test_exp, _ = jax.vmap(
      problem_instance_utils.rollout_policy,
      in_axes=(None, None, 0, None))(dynamics, expert_policy, x0s_final_test,
                                     FLAGS.horizon)

  final_test_shift = np.linalg.norm(Xs_final_test_shift[:, -1, :], axis=1)
  final_test_exp = np.linalg.norm(Xs_final_test_exp[:, -1, :], axis=1)
  final_test_delta_goal_error = np.linalg.norm(
      Xs_final_test_shift[:, -1, :] - Xs_final_test_exp[:, -1, :], axis=1)
  final_imitation_error = np.sum(
      np.linalg.norm(
          Us_final_test_shift - Us_expert_final_test_shift, axis=2), axis=1)

  logging.info('final shift goal error: %s', stats(final_test_shift))
  logging.info('expert goal error: %s', stats(final_test_exp))
  logging.info('final delta goal error: %s', stats(final_test_delta_goal_error))
  logging.info('final_imitation_error: %s', stats(final_imitation_error))

  if FLAGS.metrics_outfile is not None:
    with open(FLAGS.metrics_outfile, 'wb') as fp:
      pickle.dump(
          {
              'final_test_shift': final_test_shift,
              'final_test_exp': final_test_exp,
              'final_test_delta_goal_error': final_test_delta_goal_error,
              'final_imitation_error': final_imitation_error,
          }, fp)
  if FLAGS.config_outfile is not None:
    with open(FLAGS.config_outfile, 'wb') as fp:
      pickle.dump(get_config(), fp)
  if FLAGS.params_outfile is not None:
    with open(FLAGS.params_outfile, 'wb') as fp:
      pickle.dump(
          {
              'mixing_weight': FLAGS.alpha,
              'dagger': False,
              'policy_params': policy_params
          }, fp)


if __name__ == '__main__':
  app.run(main)
