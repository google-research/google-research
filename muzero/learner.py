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

# python3
# pylint: disable=missing-docstring
# pylint: disable=unused-argument
# pylint: disable=logging-format-interpolation
# pylint: disable=g-complex-comprehension
"""MuZero based SEED learner."""

import collections
import concurrent.futures
import copy
import os
import time

from absl import logging
import numpy as np
from seed_rl import grpc
from seed_rl.common import utils
from seed_rl.common.parametric_distribution import get_parametric_distribution_for_action_space
import tensorflow as tf

from muzero import core
from muzero import learner_config
from muzero import network


log_keys = []  # array of strings with names of values logged by compute_loss


def scale_gradient(t, scale):
  return scale * t + (1 - scale) * tf.stop_gradient(t)


def scalar_loss(yhat, y):
  return tf.square(yhat - y)


def noop_decorator(func):
  return func


def compute_pretrain_loss(config: learner_config.LearnerConfig,
                          parametric_action_distribution, agent,
                          importance_weights, *sample):

  if config.debug and np.random.rand() < 1 / 50:
    logging.info('-------------------')
    logging.info('pretrain sample:')
    logging.info(sample)
    logging.info('-------------------')

  loss, pretrain_logs = agent.pretraining_loss(sample)
  loss = loss * importance_weights  # importance sampling correction
  mean_loss = tf.math.divide_no_nan(
      tf.reduce_sum(loss), tf.reduce_sum(importance_weights))

  if config.weight_decay > 0.:
    l2_loss = config.weight_decay * sum(
        tf.nn.l2_loss(v)
        for v in agent.get_trainable_variables(pretraining=True))
  else:
    l2_loss = mean_loss * 0.

  total_loss = mean_loss + l2_loss

  del log_keys[:]
  log_values = []

  # logging

  def log(key, value):
    # this is a python op so it happens only when this tf.function is compiled
    log_keys.append(key)
    # this is a TF op
    log_values.append(value)

  log('pretrain/losses/sample', mean_loss)
  log('pretrain/losses/weight_decay', l2_loss)
  log('pretrain/losses/total', total_loss)
  if pretrain_logs is not None:
    for ptk, ptv in pretrain_logs.items():
      log('pretrain/{}'.format(ptk), tf.reduce_mean(ptv))

  return total_loss, log_values


def compute_loss(config: learner_config.LearnerConfig,
                 parametric_action_distribution, agent, importance_weights,
                 observation, history, target_value_mask, target_reward_mask,
                 target_policy_mask, target_value, target_reward,
                 target_policy):

  # initial step
  output = agent.initial_inference(observation)
  predictions = [
      core.Prediction(
          gradient_scale=1.0,
          value=output.value,
          value_logits=output.value_logits,
          reward=output.reward,
          reward_logits=output.reward_logits,
          policy_logits=output.policy_logits,
      )
  ]

  # recurrent steps
  num_recurrent_steps = history.shape[-1]
  for rstep in range(num_recurrent_steps):
    hidden_state_gradient_scale = 1.0 if rstep == 0 else 0.5
    output = agent.recurrent_inference(
        scale_gradient(output.hidden_state, hidden_state_gradient_scale),
        history[:, rstep],
    )
    predictions.append(
        core.Prediction(
            gradient_scale=1.0 / num_recurrent_steps,
            value=output.value,
            value_logits=output.value_logits,
            reward=output.reward,
            reward_logits=output.reward_logits,
            policy_logits=output.policy_logits,
        ))

  num_target_steps = target_value.shape[-1]
  assert len(predictions) == num_target_steps, (
      'There should be as many predictions ({}) as targets ({})'.format(
          len(predictions), num_target_steps))

  masks = {
      'value': target_value_mask,
      'reward': target_reward_mask,
      'policy': target_policy_mask,
      'action': target_policy_mask,
  }

  def name_to_mask(name):
    return next(k for k in masks if k in name)

  # This is more rigorous than the MuZero paper.
  gradient_scales = {
      k: tf.math.divide(1.0, tf.maximum(tf.reduce_sum(m[:, 1:], -1), 1.0))
      for k, m in masks.items()
  }
  gradient_scales = {
      k: [tf.ones_like(s)] + [s] * (num_target_steps - 1)
      for k, s in gradient_scales.items()
  }

  target_reward_encoded, target_value_encoded = (tf.reshape(
      enc.encode(tf.reshape(v, (-1,))),
      (-1, num_target_steps,
       enc.num_steps)) for enc, v in ((agent.reward_encoder, target_reward),
                                      (agent.value_encoder, target_value)))

  # Accumulators over time steps.
  accs = collections.defaultdict(list)
  for tstep, prediction in enumerate(predictions):
    accs['value_loss'].append(
        scale_gradient(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction.value_logits,
                labels=target_value_encoded[:, tstep]),
            gradient_scales['value'][tstep]))
    accs['reward_loss'].append(
        scale_gradient(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction.reward_logits,
                labels=target_reward_encoded[:, tstep]),
            gradient_scales['reward'][tstep]))
    policy_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction.policy_logits, labels=target_policy[:, tstep])
    entropy_loss = -parametric_action_distribution.entropy(
        prediction.policy_logits) * config.policy_loss_entropy_regularizer
    accs['policy_loss'].append(
        scale_gradient(policy_loss + entropy_loss,
                       gradient_scales['policy'][tstep]))

    accs['value_diff'].append(
        tf.abs(tf.squeeze(prediction.value) - target_value[:, tstep]))
    accs['reward_diff'].append(
        tf.abs(tf.squeeze(prediction.reward) - target_reward[:, tstep]))
    accs['policy_acc'].append(
        tf.keras.metrics.categorical_accuracy(
            target_policy[:, tstep],
            tf.nn.softmax(prediction.policy_logits, axis=-1)))

    accs['value'].append(tf.squeeze(prediction.value))
    accs['reward'].append(tf.squeeze(prediction.reward))
    accs['action'].append(
        tf.cast(tf.argmax(prediction.policy_logits, -1), tf.float32))

    accs['target_value'].append(target_value[:, tstep])
    accs['target_reward'].append(target_reward[:, tstep])
    accs['target_action'].append(
        tf.cast(tf.argmax(target_policy[:, tstep], -1), tf.float32))

  accs = {k: tf.stack(v, -1) * masks[name_to_mask(k)] for k, v in accs.items()}

  if config.debug and np.random.rand() < 1 / 50:
    logging.info('-------------------')
    logging.info(observation)
    for k, v in accs.items():
      logging.info('{}:\n{}'.format(k, v))
    for k, v in masks.items():
      logging.info('mask {}:\n{}'.format(k, v))
    logging.info('history:\n{}'.format(history))
    logging.info('target_policy:\n{}'.format(target_policy))
    logging.info('importance_weights:\n{}'.format(importance_weights))
    logging.info('-------------------')

  loss = accs['value_loss'] + config.reward_loss_scaling * accs[
      'reward_loss'] + config.policy_loss_scaling * accs['policy_loss']
  loss = tf.reduce_sum(loss, -1)  # aggregating over time
  loss = loss * importance_weights  # importance sampling correction
  mean_loss = tf.math.divide_no_nan(
      tf.reduce_sum(loss), tf.reduce_sum(importance_weights))

  if config.weight_decay > 0.:
    l2_loss = config.weight_decay * sum(
        tf.nn.l2_loss(v)
        for v in agent.get_trainable_variables(pretraining=False))
  else:
    l2_loss = mean_loss * 0.

  mean_loss += l2_loss

  del log_keys[:]
  log_values = []

  # logging

  def log(key, value):
    # this is a python op so it happens only when this tf.function is compiled
    log_keys.append(key)
    # this is a TF op
    log_values.append(value)

  log('losses/total', mean_loss)
  log('losses/weight_decay', l2_loss)

  sum_accs = {k: tf.reduce_sum(a, -1) for k, a in accs.items()}
  sum_masks = {
      k: tf.maximum(tf.reduce_sum(m, -1), 1.) for k, m in masks.items()
  }

  def get_mean(k):
    return tf.reduce_mean(sum_accs[k] / sum_masks[name_to_mask(k)])

  log('prediction/value', get_mean('value'))
  log('prediction/reward', get_mean('reward'))
  log('prediction/policy', get_mean('action'))

  log('target/value', get_mean('target_value'))
  log('target/reward', get_mean('target_reward'))
  log('target/policy', get_mean('target_action'))

  log('losses/value', tf.reduce_mean(sum_accs['value_loss']))
  log('losses/reward', tf.reduce_mean(sum_accs['reward_loss']))
  log('losses/policy', tf.reduce_mean(sum_accs['policy_loss']))

  log('accuracy/value', -get_mean('value_diff'))
  log('accuracy/reward', -get_mean('reward_diff'))
  log('accuracy/policy', get_mean('policy_acc'))

  return mean_loss, log_values


def validate_config():
  pass


def make_spec_from_gym_space(space, name):
  if space.dtype is not None:
    specs = tf.TensorSpec(space.shape, space.dtype, name)
  else:
    # This is a tuple space
    specs = tuple(
        make_spec_from_gym_space(s, '{}_{}'.format(name, idx))
        for idx, s in enumerate(space.spaces))
  return specs


def learner_loop(env_descriptor,
                 create_agent_fn,
                 create_optimizer_fn,
                 config: learner_config.LearnerConfig,
                 mzconfig,
                 pretraining=False):
  """Main learner loop.

  Args:
    env_descriptor: An instance of utils.EnvironmentDescriptor.
    create_agent_fn: Function that must create a new tf.Module with the neural
      network that outputs actions and new agent state given the environment
      observations and previous agent state. See dmlab.agents.ImpalaDeep for an
      example. The factory function takes as input the environment descriptor
      and a parametric distribution over actions.
    create_optimizer_fn: Function that takes the final iteration as argument and
      must return a tf.keras.optimizers.Optimizer and a
      tf.keras.optimizers.schedules.LearningRateSchedule.
    config: A LearnerConfig object.
    mzconfig: A MuZeroConfig object.
    pretraining: Do pretraining.
  """
  logging.info('Starting learner loop')
  validate_config()
  settings = utils.init_learner(config.num_training_tpus)
  strategy, inference_devices, training_strategy, encode, decode = settings
  tf_function = noop_decorator if config.debug else tf.function
  parametric_action_distribution = get_parametric_distribution_for_action_space(
      env_descriptor.action_space)

  observation_specs = make_spec_from_gym_space(env_descriptor.observation_space,
                                               'observation')
  action_specs = make_spec_from_gym_space(env_descriptor.action_space, 'action')

  if pretraining:
    assert env_descriptor.pretraining_space is not None, (
        'Must define a pretraining space')
    pretraining_specs = make_spec_from_gym_space(
        env_descriptor.pretraining_space, 'pretraining')

  # Initialize agent and variables.
  with strategy.scope():
    agent = create_agent_fn(env_descriptor, parametric_action_distribution)
  initial_agent_state = agent.initial_state(1)
  if config.debug:
    logging.info('initial state:\n{}'.format(initial_agent_state))

  agent_state_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)

  zero_observation = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), observation_specs)
  zero_action = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), action_specs)

  zero_initial_args = [encode(zero_observation)]
  zero_recurrent_args = [encode(initial_agent_state), encode(zero_action)]
  if config.debug:
    logging.info('zero initial args:\n{}'.format(zero_initial_args))
    logging.info('zero recurrent args:\n{}'.format(zero_recurrent_args))

  if pretraining:
    zero_pretraining = tf.nest.map_structure(
        lambda s: tf.zeros([1] + list(s.shape), s.dtype), pretraining_specs)
    zero_pretraining_args = [encode(zero_pretraining)]
    logging.info('zero pretraining args:\n{}'.format(zero_pretraining_args))
  else:
    zero_pretraining_args = None

  with strategy.scope():

    def create_variables(initial_args, recurrent_args, pretraining_args):
      agent.initial_inference(*map(decode, initial_args))
      agent.recurrent_inference(*map(decode, recurrent_args))
      if pretraining_args is not None:
        agent.pretraining_loss(*map(decode, pretraining_args))

    # This complicates BatchNormalization, can't use it.
    create_variables(zero_initial_args, zero_recurrent_args,
                     zero_pretraining_args)

  with strategy.scope():
    # Create optimizer.
    optimizer, learning_rate_fn = create_optimizer_fn(config.total_iterations)

    # pylint: disable=protected-access
    iterations = optimizer.iterations
    optimizer._create_hypers()
    optimizer._create_slots(
        agent.get_trainable_variables(pretraining=pretraining))
    # pylint: enable=protected-access

  with strategy.scope():
    # ON_READ causes the replicated variable to act as independent variables for
    # each replica.
    temp_grads = [
        tf.Variable(
            tf.zeros_like(v),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            name='temp_grad_{}'.format(v.name),
        ) for v in agent.get_trainable_variables(pretraining=pretraining)
    ]

  logging.info('--------------------------')
  logging.info('TRAINABLE VARIABLES')
  for v in agent.get_trainable_variables(pretraining=pretraining):
    logging.info('{}: {} | {}'.format(v.name, v.shape, v.dtype))
  logging.info('--------------------------')

  @tf_function
  def _compute_loss(*args, **kwargs):
    if pretraining:
      return compute_pretrain_loss(config, *args, **kwargs)
    else:
      return compute_loss(config, *args, **kwargs)

  @tf_function
  def minimize(iterator):
    data = next(iterator)

    @tf_function
    def compute_gradients(args):
      args = tf.nest.pack_sequence_as(weighted_replay_buffer_specs,
                                      decode(args, data))
      with tf.GradientTape() as tape:
        loss, logs = _compute_loss(parametric_action_distribution, agent, *args)
      grads = tape.gradient(
          loss, agent.get_trainable_variables(pretraining=pretraining))
      for t, g in zip(temp_grads, grads):
        t.assign(g if g is not None else tf.zeros_like(t))
      return loss, logs

    loss, logs = training_strategy.run(compute_gradients, (data,))
    loss = training_strategy.experimental_local_results(loss)[0]
    logs = training_strategy.experimental_local_results(logs)

    @tf_function
    def apply_gradients(_):
      grads = temp_grads
      if config.gradient_norm_clip > 0.:
        grads, _ = tf.clip_by_global_norm(grads, config.gradient_norm_clip)
      optimizer.apply_gradients(
          zip(grads, agent.get_trainable_variables(pretraining=pretraining)))

    strategy.run(apply_gradients, (loss,))

    return logs

  # Logging.
  logdir = os.path.join(config.logdir, 'learner')
  summary_writer = tf.summary.create_file_writer(
      logdir,
      flush_millis=config.flush_learner_log_every_n_s * 1000,
      max_queue=int(1E6))

  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
  manager = tf.train.CheckpointManager(
      ckpt, logdir, max_to_keep=1, keep_checkpoint_every_n_hours=6)

  # Continuing a run from an intermediate checkpoint.  On this path, we do not
  # need to read `init_checkpoint`.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
    last_ckpt_time = time.time()

    # Also properly reset iterations.
    iterations = optimizer.iterations
  else:
    last_ckpt_time = 0  # Force checkpointing of the initial model.
    # If there is a checkpoint from pre-training specified, load it now.
    # Note that we only need to do this if we are not already restoring a
    # checkpoint from the actual training.
    if config.init_checkpoint is not None:
      logging.info('Loading initial checkpoint from %s ...',
                   config.init_checkpoint)
      # We don't want to restore the optimizer from pretraining
      ckpt_without_optimizer = tf.train.Checkpoint(agent=agent)
      # Loading checkpoints from independent pre-training might miss, for
      # example, optimizer weights (or have used different optimizers), and
      # might also not have fully instantiated all network parts (e.g. the
      # "core"-recurrence).
      # We still want to catch cases where nothing at all matches, but can not
      # do anything stricter here.
      ckpt_without_optimizer.restore(
          config.init_checkpoint).assert_nontrivial_match()
      logging.info('Finished loading the initial checkpoint.')

  server = grpc.Server([config.server_address])

  num_target_steps = mzconfig.num_unroll_steps + 1
  target_specs = (
      tf.TensorSpec([num_target_steps], tf.float32, 'value_mask'),
      tf.TensorSpec([num_target_steps], tf.float32, 'reward_mask'),
      tf.TensorSpec([num_target_steps], tf.float32, 'policy_mask'),
      tf.TensorSpec([num_target_steps], tf.float32, 'value'),
      tf.TensorSpec([num_target_steps], tf.float32, 'reward'),
      tf.TensorSpec([num_target_steps, env_descriptor.action_space.n],
                    tf.float32, 'policy'),
  )

  if pretraining:
    replay_buffer_specs = pretraining_specs
  else:
    replay_buffer_specs = (
        observation_specs,
        tf.TensorSpec(
            env_descriptor.action_space.shape + (mzconfig.num_unroll_steps,),
            env_descriptor.action_space.dtype, 'history'),
        *target_specs,
    )

  weighted_replay_buffer_specs = (
      tf.TensorSpec([], tf.float32, 'importance_weights'), *replay_buffer_specs)

  episode_stat_specs = (
      tf.TensorSpec([], tf.string, 'summary_name'),
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.int64, 'episode_length'),
  )
  if env_descriptor.extras:
    episode_stat_specs += tuple(
        tf.TensorSpec([], stat[1], stat[0])
        for stat in env_descriptor.extras.get('learner_stats', []))

  replay_buffer_size = config.replay_buffer_size
  replay_buffer = utils.PrioritizedReplay(
      replay_buffer_size,
      replay_buffer_specs,
      config.importance_sampling_exponent,
  )

  replay_queue_specs = (
      tf.TensorSpec([], tf.float32, 'priority'),
      *replay_buffer_specs,
  )
  replay_queue_size = config.replay_queue_size
  replay_buffer_queue = utils.StructuredFIFOQueue(replay_queue_size,
                                                  replay_queue_specs)

  episode_stat_queue = utils.StructuredFIFOQueue(-1, episode_stat_specs)

  def get_add_batch_size(batch_size):

    def add_batch_size(ts):
      return tf.TensorSpec([batch_size] + list(ts.shape), ts.dtype, ts.name)

    return add_batch_size

  def make_inference_fn(inference_device, inference_fn, *args):

    args = encode(args)

    def device_specific_inference_fn():
      with tf.device(inference_device):

        @tf_function
        def agent_inference(*args):
          return inference_fn(*decode(args), training=False)

        return agent_inference(*args)

    return device_specific_inference_fn

  initial_inference_specs = (observation_specs,)

  def make_initial_inference_fn(inference_device):

    @tf.function(
        input_signature=tf.nest.map_structure(
            get_add_batch_size(config.initial_inference_batch_size),
            initial_inference_specs))
    def initial_inference(observation):
      return make_inference_fn(inference_device, agent.initial_inference,
                               observation)()

    return initial_inference

  recurrent_inference_specs = (
      agent_state_specs,
      action_specs,
  )

  def make_recurrent_inference_fn(inference_device):

    @tf.function(
        input_signature=tf.nest.map_structure(
            get_add_batch_size(config.recurrent_inference_batch_size),
            recurrent_inference_specs))
    def recurrent_inference(hidden_state, action):
      return make_inference_fn(inference_device, agent.recurrent_inference,
                               hidden_state, action)()

    return recurrent_inference

  @tf.function(
      input_signature=tf.nest.map_structure(
          get_add_batch_size(config.batch_size), replay_queue_specs))
  def add_to_replay_buffer(*batch):
    queue_size = replay_buffer_queue.size()
    num_free = replay_queue_size - queue_size
    if not config.replay_queue_block and num_free < config.recurrent_inference_batch_size:
      replay_buffer_queue.dequeue_many(config.recurrent_inference_batch_size)
    replay_buffer_queue.enqueue_many(batch)

  @tf.function(input_signature=episode_stat_specs)
  def add_to_reward_queue(*stats):
    episode_stat_queue.enqueue(stats)

  @tf.function(input_signature=[])
  def learning_iteration():
    return optimizer.iterations

  with strategy.scope():
    server.bind([make_initial_inference_fn(d) for d in inference_devices])
    server.bind([make_recurrent_inference_fn(d) for d in inference_devices])
    server.bind(add_to_replay_buffer)
    server.bind(add_to_reward_queue)
    server.bind(learning_iteration)
  server.start()

  @tf_function
  def dequeue(ctx):

    while tf.constant(True):

      num_dequeues = config.learner_skip + 1
      if num_dequeues < 1:
        queue_size = replay_buffer_queue.size()
        num_dequeues = tf.maximum(queue_size // config.batch_size - 1,
                                  tf.ones_like(queue_size))
      for _ in tf.range(num_dequeues):
        batch = replay_buffer_queue.dequeue_many(config.batch_size)
        priorities, *samples = batch
        replay_buffer.insert(tuple(samples), priorities)

      if replay_buffer.num_inserted >= replay_buffer_size:
        break

      tf.print(
          'waiting for replay buffer to fill. Status:',
          replay_buffer.num_inserted,
          ' / ',
          replay_buffer_size,
      )

    indices, weights, replays = replay_buffer.sample(
        ctx.get_per_replica_batch_size(config.batch_size),
        config.priority_sampling_exponent)
    if config.replay_buffer_update_priority_after_sampling_value >= 0.:
      replay_buffer.update_priorities(
          indices,
          tf.convert_to_tensor(
              np.ones(indices.shape) *
              config.replay_buffer_update_priority_after_sampling_value,
              dtype=tf.float32))

    data = (weights, *replays)
    data = tuple(map(encode, data))

    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and
    # repack.
    return tf.nest.flatten(data)

  def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)
    return dataset.map(
        lambda _: dequeue(ctx), num_parallel_calls=ctx.num_replicas_in_sync)

  dataset = training_strategy.experimental_distribute_datasets_from_function(
      dataset_fn)
  it = iter(dataset)

  # Execute learning and track performance.
  with summary_writer.as_default(), \
       concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    log_future = executor.submit(lambda: None)  # No-op future.
    last_iterations = iterations
    last_log_time = time.time()
    values_to_log = collections.defaultdict(lambda: [])
    while iterations < config.total_iterations:
      tf.summary.experimental.set_step(iterations)

      # Save checkpoint.
      current_time = time.time()
      if current_time - last_ckpt_time >= config.save_checkpoint_secs:
        manager.save()
        if config.export_agent:
          # We also export the agent as a SavedModel to be used for inference.
          saved_model_dir = os.path.join(logdir, 'saved_model')
          network.export_agent_for_initial_inference(
              agent=agent,
              model_dir=os.path.join(saved_model_dir, 'initial_inference'))
          network.export_agent_for_recurrent_inference(
              agent=agent,
              model_dir=os.path.join(saved_model_dir, 'recurrent_inference'))
        last_ckpt_time = current_time

      def log(iterations):
        """Logs batch and episodes summaries."""
        nonlocal last_iterations, last_log_time
        summary_writer.set_as_default()
        tf.summary.experimental.set_step(iterations)

        # log data from the current minibatch
        for key, values in copy.deepcopy(values_to_log).items():
          if values:
            tf.summary.scalar(key, values[-1])  # could also take mean
        values_to_log.clear()
        tf.summary.scalar('learning_rate', learning_rate_fn(iterations))
        tf.summary.scalar('replay_queue_size', replay_buffer_queue.size())
        stats = episode_stat_queue.dequeue_many(episode_stat_queue.size())

        summary_name_idx = [spec.name for spec in episode_stat_specs
                           ].index('summary_name')
        summary_name_stats = stats[summary_name_idx]
        unique_summary_names, unique_summary_name_idx = tf.unique(
            summary_name_stats)

        def log_mean_value(values, label):
          mean_value = tf.reduce_mean(tf.cast(values, tf.float32))
          tf.summary.scalar(label, mean_value)


        for stat, stat_spec in zip(stats, episode_stat_specs):
          if stat_spec.name == 'summary_name' or len(stat) <= 0:
            continue

          for idx, summary_name in enumerate(unique_summary_names):
            add_to_summary = unique_summary_name_idx == idx
            stat_masked = tf.boolean_mask(stat, add_to_summary)
            label = f'{summary_name.numpy().decode()}/mean_{stat_spec.name}'
            if len(stat_masked) > 0:  # pylint: disable=g-explicit-length-test
              log_mean_value(stat_masked, label=label)

      logs = minimize(it)

      if (config.enable_learner_logging == 1 and
          iterations % config.log_frequency == 0):
        for per_replica_logs in logs:
          assert len(log_keys) == len(per_replica_logs)
          for key, value in zip(log_keys, per_replica_logs):
            try:
              values_to_log[key].append(value.numpy())
            except AttributeError:
              values_to_log[key].extend(
                  x.numpy()
                  for x in training_strategy.experimental_local_results(value))

        log_future.result()  # Raise exception if any occurred in logging.
        log_future = executor.submit(log, iterations)

  manager.save()
  server.shutdown()
