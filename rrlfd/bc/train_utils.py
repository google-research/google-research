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

"""Shared training utilities."""

import os
import pickle
from typing import Dict, Optional

from absl import flags
import gym
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from tensorflow.io import gfile
from rrlfd.bc import bc_agent
from rrlfd.bc import eval_loop
from rrlfd.bc import pickle_dataset

tfd = tfp.distributions

flags.DEFINE_integer('split_seed', None, 'Seed to split train and val data.')
flags.DEFINE_boolean('dataset_in_ram', True,
                     'If True, load dataset into memory. Else read from disk.')
flags.DEFINE_boolean('decompress_demos_once', True,
                     'If True, keep decompressed frames in memory. Else '
                     'decompress on demand.')

flags.DEFINE_integer('shuffle_size', 50_000, 'Buffer size for shuffling.')

FLAGS = flags.FLAGS


def make_environment(domain, task, use_egl=False, render=False,
                     image_size=None):
  """Initialize gym environment."""
  if domain == 'mime':
    egl_str = '-EGL' if use_egl else ''
    env = gym.make(f'UR5{egl_str}-{task}CamEnv-v0')
    scene = env.unwrapped.scene
    scene.renders(render)
    if image_size is not None:
      env.env._cam_resolution = (image_size, image_size)  # pylint: disable=protected-access
  elif domain == 'adroit':
    env = gym.make(f'visual-{task}-v0')
    if image_size is not None:
      env.env.im_size = image_size
  else:
    raise NotImplementedError('Domain', domain, 'not defined.')
  return env


def get_summary_dir_for_ckpt_dir(ckpt_dir, network):
  base_dir = ckpt_dir
  job_ids = []
  while os.path.basename(base_dir) != network:
    job_ids = [os.path.basename(base_dir)] + job_ids
    base_dir = os.path.dirname(base_dir)
  summary_dir = os.path.join(base_dir, 'tb', *job_ids)
  return summary_dir


def reset_action_stats(demos_file, max_demos_to_load, max_demo_length,
                       val_size, val_full_episodes, split_seed, agent,
                       split_dir=None):
  """Set an agent's normalization statistics based on a demostration dataset."""
  dataset = pickle_dataset.DemoReader(
      path=demos_file,
      max_to_load=max_demos_to_load,
      max_demo_length=max_demo_length,
      load_in_memory=FLAGS.dataset_in_ram,
      agent=agent)
  episode_train_split = None
  if split_dir is not None:
    episode_split_path = os.path.join(split_dir, 'episode_train_split.pkl')
    if gfile.exists(episode_split_path):
      with gfile.GFile(episode_split_path, 'rb') as f:
        episode_train_split = pickle.load(f)
  split_seed = int(FLAGS.split_seed) if FLAGS.split_seed is not None else None
  dataset.create_split(
      val_size, val_full_episodes, seed=split_seed,
      episode_train_split=episode_train_split)
  dataset.compute_dataset_stats()
  # TODO(minttu): del dataset?
  agent.reset_action_stats(dataset.action_mean, dataset.action_std)
  agent.reset_observation_stats(dataset.signal_mean, dataset.signal_std)


def prepare_demos(
    demos_file, input_type, max_demos_to_load, max_demo_length, augment_frames,
    agent, split_dir, val_size, val_full_episodes, skip=0,
    reset_agent_stats=True):
  """Load and preprocess a demonstration dataset."""
  dataset = pickle_dataset.DemoReader(
      path=demos_file,
      input_type=input_type,
      max_to_load=max_demos_to_load,
      max_demo_length=max_demo_length,
      augment_frames=augment_frames,
      load_in_memory=FLAGS.dataset_in_ram,
      decompress_once=FLAGS.decompress_demos_once,
      agent=agent,
      skip=skip)
  episode_train_split = None
  if split_dir is not None:
    episode_split_path = os.path.join(split_dir, 'episode_train_split.pkl')
    if gfile.exists(episode_split_path):
      with gfile.GFile(episode_split_path, 'rb') as f:
        episode_train_split = pickle.load(f)
  split_seed = int(FLAGS.split_seed) if FLAGS.split_seed is not None else None
  dataset.create_split(
      val_size, val_full_episodes, seed=split_seed,
      episode_train_split=episode_train_split)
  if agent is not None and reset_agent_stats:
    agent.reset_action_stats(dataset.action_mean, dataset.action_std)
    agent.reset_observation_stats(dataset.signal_mean, dataset.signal_std)

  if FLAGS.policy_init_path is not None:
    np.random.seed(0)
    dummy_input = np.random.rand(128, 3, 128, 128, 3)
    dummy_scalars = np.random.rand(128, agent.robot_info_dim)
    norm_input = agent.img_obs_space.normalize(dummy_input)
    norm_scalars = agent.signals_obs_space.normalize_flat(dummy_scalars)
    print(agent.visible_state_features)
    unnorm_pred = agent.network.call(norm_input, norm_scalars, interrupt=False)
    norm_pred = agent.action_space.denormalize(unnorm_pred)
    print('norm pred\n', norm_pred)
  return dataset


def nll(y, pred, log_std):
  z = (y - pred) / tf.exp(log_std)
  ll = -0.5 * z ** 2 - log_std - 0.5 * tf.math.log(2 * np.pi)
  return -tf.reduce_sum(ll)


def predict(model, x_img, x_scalars, training):
  """Pass the relevant arguments to model call."""
  if isinstance(model.network, snt.Sequential):
    # Assumes no layers need training mode flag.
    pred = model(x_img)
    # TODO(minttu): NLL for MultivariateNormalDiag, with constant learned stds.
    if isinstance(pred, tfd.MultivariateNormalDiag):
      pred = pred.mean()
  else:
    pred = model(x_img, input_scalars=x_scalars, training=training)
  return pred


def nll_ce_loss(model, x_img, x_scalars, y, l2_weight, training):
  """Compute weighted combination of NLL and cross-entropy loss terms."""
  pred = model(x_img, input_scalars=x_scalars, training=training)
  ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  log_std = model.log_std

  idx_net = np.arange(0, pred.shape[1], model.action_pred_dim)
  idx_action = np.arange(0, y.shape[1], model.action_target_dim)
  sum_ce = 0
  sum_nll = 0
  for idx_n, idx_a in zip(idx_net, idx_action):
    gripper_state = y[:, idx_a] < 0  # 1 if closed
    # First logit: open, second: close.
    sum_ce += ce(y_true=gripper_state,
                 y_pred=pred[:, idx_n:(idx_n + 2)])
    sum_nll += nll(y[:, (idx_a + 1):(idx_a + model.action_target_dim)],
                   pred[:, (idx_n + 2):(idx_n + model.action_pred_dim)],
                   log_std[(idx_a + 1):(idx_n + model.action_target_dim)])
  loss = l2_weight * sum_nll + (1 - l2_weight) * sum_ce
  return loss, sum_nll, sum_ce


def nll_loss(model, x_img, x_scalars, y, training):
  """Compute negative log-likelihood loss for a batch of actions."""
  pred = model(x_img, input_scalars=x_scalars, training=training)
  # TODO(minttu): Sum over action predictions if predicting multiple at once.
  pred_denorm = model.action_space.denormalize(pred)
  y_denorm = model.action_space.denormalize(y)
  pred = pred_denorm
  y = y_denorm
  log_std = model.log_std
  z = (y - pred) / tf.exp(log_std)
  ll = -0.5 * z ** 2 - log_std - 0.5 * tf.math.log(2 * np.pi)
  # Dimension 0 is for batch.
  nll_sum = -tf.reduce_sum(ll, axis=1)
  batch_nll = tf.reduce_mean(nll_sum)
  return batch_nll


def l2_ce_loss(model, x_img, x_scalars, y, l2_weight, training):
  """Compute weighted cross-entropy and regression loss."""
  pred = model(x_img, input_scalars=x_scalars, training=training)
  ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  l2 = tf.keras.losses.MeanSquaredError()
  idx_net = np.arange(0, pred.shape[1], model.action_pred_dim)
  idx_action = np.arange(0, y.shape[1], model.action_target_dim)
  sum_ce = 0
  sum_l2 = 0
  for idx_n, idx_a in zip(idx_net, idx_action):
    gripper_state = y[:, idx_a] < 0  # 1 if closed
    # First logit: open, second: close.
    sum_ce += ce(y_true=gripper_state,
                 y_pred=pred[:, idx_n:(idx_n + 2)])
    sum_l2 += l2(y_true=y[:, (idx_a + 1):(idx_a + model.action_target_dim)],
                 y_pred=pred[:, (idx_n + 2):(idx_n + model.action_pred_dim)])
  loss = l2_weight * sum_l2 + (1 - l2_weight) * sum_ce
  return loss, sum_l2, sum_ce


def l2_loss(model, x_img, x_scalars, y, training):
  """Compute regression loss."""
  pred = predict(model, x_img, x_scalars, training)
  l2 = tf.keras.losses.MeanSquaredError()
  loss = l2(y_true=y, y_pred=pred)
  return loss


def eval_batch(model, img_batch, feats_batch, act_batch, binary_grip_action,
               loss_fn, l2_weight, training=False):
  """Compute loss for a batch and return a breakdown of loss terms."""
  loss_breakdown = {}
  if binary_grip_action:
    if loss_fn == 'l2':
      loss, l2, ce = l2_ce_loss(
          model, img_batch, feats_batch, act_batch, l2_weight, training)
      loss_breakdown['l2'] = l2
    else:
      loss, nll_value, ce = nll_ce_loss(
          model, img_batch, feats_batch, act_batch, l2_weight, training)
      loss_breakdown['nll'] = nll_value
    loss_breakdown['ce'] = ce
    loss_breakdown['loss'] = loss
  else:
    if loss_fn == 'l2':
      loss = l2_loss(model, img_batch, feats_batch, act_batch, training)
    else:
      loss = nll_loss(model, img_batch, feats_batch, act_batch, training)
    loss_breakdown['loss'] = loss
  return loss_breakdown


def train_step(model, optimizer, img_batch, feats_batch, act_batch,
               binary_grip_action, loss_fn, l2_weight):
  """Apply a train step on model parameters."""
  with tf.GradientTape() as tape:
    loss_breakdown = eval_batch(
        model, img_batch, feats_batch, act_batch, binary_grip_action, loss_fn,
        l2_weight, training=True)
    loss = loss_breakdown['loss']
  grads = tape.gradient(loss, model.network.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.network.trainable_variables))
  model.network.log_std.assign(tf.math.maximum(model.network.log_std, -3))
  return loss_breakdown


def eval_on_dataset(data, agent, loss_fn, l2_weight):
  """Evaluate agent on data using loss_fn and l2 weight (if applicable)."""
  val_losses = {}
  val_data_size = 0
  # Data should generate
  # (image, scalars, actions, [possibly other items]) tuples.
  for batches in data:
    img_batch = batches[0]
    feats_batch = batches[1]
    act_batch = batches[2]
    val_loss_breakdown = eval_batch(
        agent, img_batch, feats_batch, act_batch, agent.binary_grip_action,
        loss_fn, l2_weight)
    for k, v in val_loss_breakdown.items():
      if k not in val_losses:
        val_losses[k] = v.numpy()
      else:
        val_losses[k] = (
            (val_losses[k] * val_data_size + v.numpy() * len(img_batch))
            / (val_data_size + len(img_batch)))
    val_data_size += len(img_batch)
  return val_losses


def train_on_dataset(
    dataset, agent, optimizer, batch_size, num_epochs, loss_fn, l2_weight,
    ckpt_path, summary_dir, epochs_to_eval, shuffle_seed):
  """Training agent on dataset with behavioural cloning."""
  train_set = tf.data.Dataset.from_generator(
      dataset.generate_train_timestep,
      (tf.float32, tf.float32, tf.float32, tf.int64, tf.int64))
  val_set = tf.data.Dataset.from_generator(
      dataset.generate_val_timestep,
      (tf.float32, tf.float32, tf.float32, tf.int64, tf.int64))
  train_set = train_set.shuffle(
      FLAGS.shuffle_size, seed=shuffle_seed, reshuffle_each_iteration=True)
  train_set = train_set.batch(batch_size)
  val_set = val_set.batch(batch_size)

  best_val_loss = np.inf
  best_val_loss_epoch = -1
  summary_writer = None
  if summary_dir is not None:
    summary_writer = tf.summary.create_file_writer(summary_dir)
  binary_grip_action = agent.binary_grip_action

  # Evaluate once before training.
  train_losses = eval_on_dataset(train_set, agent, loss_fn, l2_weight)
  print('train', '   '.join(
      [f'{k}: {v:.6f}' for k, v in sorted(train_losses.items())]))
  val_losses = eval_on_dataset(val_set, agent, loss_fn, l2_weight)
  print('val  ', '   '.join(
      [f'{k}: {v:.6f}' for k, v in sorted(val_losses.items())]))
  agent.save(ckpt_path + '_init')
  if summary_writer is not None:
    with summary_writer.as_default():
      for k, v in train_losses.items():
        tf.summary.scalar(f'train_{k}', v, step=0)
      for k, v in val_losses.items():
        tf.summary.scalar(f'val_{k}', v, step=0)
      summary_writer.flush()

  for e in np.arange(num_epochs):
    print('\nepoch', e + 1)
    train_losses = {}
    pbar = tqdm(train_set)
    for b, batch in enumerate(pbar):
      img_batch, feats_batch, act_batch, unused_demo_idxs, unused_ts = batch
      loss_breakdown = train_step(
          agent, optimizer, img_batch, feats_batch, act_batch,
          binary_grip_action, loss_fn, l2_weight)
      for k, v in loss_breakdown.items():
        if k not in train_losses:
          train_losses[k] = v.numpy()
        else:
          train_losses[k] = (train_losses[k] * b + v.numpy()) / (b + 1)
      pbar.set_description(
          'train ' + '   '.join(
              [f'{k}: {v:.6f}' for k, v in sorted(train_losses.items())]))

    if loss_fn == 'nll':
      print('e', e, 'log std', agent.log_std)
      print('e', e, 'std', np.exp(agent.log_std))
    val_losses = eval_on_dataset(val_set, agent, loss_fn, l2_weight)
    print('val  ', '   '.join(
        [f'{k}: {v:.6f}' for k, v in sorted(val_losses.items())]))
    if 'loss' not in val_losses or val_losses['loss'] < best_val_loss:
      if 'loss' in val_losses:
        print(f'val loss improved from {best_val_loss:.6f} to ',
              f'{val_losses["loss"]:.6f}')
        best_val_loss = val_losses['loss']
      best_val_loss_epoch = e + 1
      if ckpt_path is not None:
        agent.save(ckpt_path)
        print('Saved weights to', ckpt_path)
    else:
      print(f'val loss did not improve from {best_val_loss:.6f} '
            f'(epoch {best_val_loss_epoch})')
    if ((e + 1) % 50 == 0 or (e + 1) == num_epochs) and ckpt_path is not None:
      agent.save(ckpt_path + '_' + str(e + 1))
      print('Saved weights to', ckpt_path + '_' + str(e + 1))
    if e + 1 in epochs_to_eval and ckpt_path is not None:
      print('Copying ckpt to', ckpt_path + f'_best_of_{e + 1}')
      gfile.Copy(ckpt_path + '.index', ckpt_path + f'_best_of_{e + 1}.index')
      for path in gfile.Glob(ckpt_path + '.data*'):
        new_filename = (
            os.path.basename(path).replace('.data', f'_best_of_{e + 1}.data'))
        gfile.Copy(path, os.path.join(os.path.dirname(path), new_filename))
      with gfile.GFile(
          os.path.join(os.path.dirname(ckpt_path),
                       f'best_epoch_of_{e + 1}'), 'w') as f:
        f.write(str(best_val_loss_epoch))

    if summary_writer is not None:
      with summary_writer.as_default():
        for k, v in train_losses.items():
          tf.summary.scalar(f'train_{k}', v, step=e + 1)
        for k, v in val_losses.items():
          tf.summary.scalar(f'val_{k}', v, step=e + 1)
        summary_writer.flush()

  print(f'\nBest epoch: {best_val_loss_epoch}, val_loss: {best_val_loss:.8f}')
  if ckpt_path is not None:
    with gfile.GFile(
        os.path.join(os.path.dirname(ckpt_path), 'best_epoch'), 'w') as f:
      f.write(str(best_val_loss_epoch))
  return best_val_loss_epoch


def train(dataset, agent, ckpt_dir, optimizer_type, learning_rate, batch_size,
          num_epochs, loss_fn, l2_weight, summary_dir=None, epochs_to_eval=(),
          shuffle_seed=None):
  """Train agent on dataset."""
  # print(agent.network.summary())
  if optimizer_type == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  else:
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

  ckpt_path = None
  if ckpt_dir is not None:
    ckpt_path = os.path.join(ckpt_dir, 'ckpt')
    if not gfile.exists(os.path.dirname(ckpt_path)):
      gfile.makedirs(os.path.dirname(ckpt_path))
    # agent.save(ckpt_path + '_init')

  best_epoch = (
      train_on_dataset(
          dataset, agent, optimizer, batch_size, num_epochs, loss_fn, l2_weight,
          ckpt_path, summary_dir, epochs_to_eval, shuffle_seed))

  # Restore to best epoch.
  if ckpt_dir is not None:
    agent.load(ckpt_path)
  return best_epoch


def eval_on_valset(dataset, agent, loss_fn, l2_weight):
  val_set = tf.data.Dataset.from_generator(
      dataset.generate_val_timestep,
      (tf.float32, tf.float32, tf.float32, tf.int64, tf.int64))
  val_set = val_set.batch(FLAGS.batch_size)
  val_losses = eval_on_dataset(val_set, agent, loss_fn, l2_weight)
  print('val  ', '   '.join(
      [f'{k}: {v:.6f}' for k, v in sorted(val_losses.items())]))
  return val_losses


def make_summary_writer(summary_dir):
  summary_writer = None
  if summary_dir is not None:
    summary_writer = tf.summary.create_file_writer(summary_dir)
  return summary_writer


def set_job_id():
  """Format job id, reflecting whether or not XManager is used."""
  job_id = FLAGS.job_id
  # If running with XM and job_id flag is not set.
  if not job_id and FLAGS.xm_xid != -1:
    job_id = f'{FLAGS.xm_xid}-{FLAGS.xm_wid}'
  # If running with XM and job_id flag is set but doesn't have '-': add XM wid.
  elif '-' not in job_id and FLAGS.xm_xid != -1:
    job_id = f'{job_id}-{FLAGS.xm_wid}'
  return job_id


# TODO(minttu): create eval_utils file with the below functions
def set_eval_path(ckpt_dir, custom_eval_id, ckpt=None):
  """Add checkpoint information and a custom eval id to evalution log path."""
  ckpt_id = ckpt.replace('ckpt_', '').replace('ckpt', '') if ckpt else ''
  eval_id = (
      f'{custom_eval_id}_{ckpt_id}' if custom_eval_id and ckpt_id
      else custom_eval_id + ckpt_id)
  if eval_id:
    eval_id = '_' + eval_id
  increment_str = 'i' if FLAGS.increment_eval_seed else ''
  eval_str = (
      f'{FLAGS.eval_task}_s{FLAGS.eval_seed}{increment_str}'
      f'_e{FLAGS.num_eval_episodes}{eval_id}')
  if FLAGS.stop_if_stuck:
    eval_str += '_s'
  eval_path = os.path.join(ckpt_dir, f'eval{eval_str}')
  return eval_path


def set_summary_key(custom_eval_id, ckpt=None):
  ckpt_id = ckpt.replace('ckpt_', '').replace('ckpt', '') if ckpt else ''
  eval_id = (
      f'{custom_eval_id}_{ckpt_id}' if custom_eval_id and ckpt_id
      else custom_eval_id + ckpt_id)
  if eval_id:
    eval_id = '_' + eval_id
  increment_str = 'i' if FLAGS.increment_eval_seed else ''
  summary_key = f'{FLAGS.eval_task}_s{FLAGS.eval_seed}{increment_str}{eval_id}'
  return summary_key


def set_eval_paths(ckpt_dir, ckpt, custom_eval_id):
  """Set paths for evaluation and TensorBoard summaries."""
  eval_path, summary_key, best_epoch = None, None, None
  if ckpt is not None:
    best_epoch_path = (
        ckpt.replace('ckpt_best_of_', 'best_epoch_of_')
        .replace('ckpt', 'best_epoch'))
    best_epoch_path = os.path.join(ckpt_dir, best_epoch_path)
    if gfile.exists(best_epoch_path):
      with gfile.GFile(best_epoch_path) as f:
        best_epoch = int(f.read())
    else:
      best_epoch = int(ckpt.replace('ckpt_', ''))
    print('best epoch:', best_epoch)
    eval_path = set_eval_path(ckpt_dir, custom_eval_id, ckpt)
    summary_key = set_summary_key(custom_eval_id, ckpt)
  return eval_path, summary_key, best_epoch


def evaluate_checkpoint(
    ckpt,
    ckpt_dir,
    agent,
    env,
    num_eval_episodes,
    epoch_to_success_rates,
    summary_writer = None,
    test_dataset = None):
  """Load weights for agent from a given ckpt and log to appropriate paths."""
  best_epoch = None
  if ckpt is not None:
    ckpt_to_load = os.path.join(ckpt_dir, ckpt)
    print('Loading from', ckpt_to_load)
    agent.load(ckpt_to_load)
  eval_path, summary_key, best_epoch = set_eval_paths(
      ckpt_dir, ckpt, FLAGS.eval_id)
  num_videos_to_save = FLAGS.eval_episodes_to_save

  if best_epoch is not None and best_epoch in epoch_to_success_rates:
    print('Reusing evaluation for epoch', best_epoch)
    success_rates = epoch_to_success_rates[best_epoch]
    for k, v in success_rates.items():
      eval_loop.log_success_rate(k, v, summary_writer, summary_key)
    # TODO(minttu): Might want to copy success path etc.
  else:
    if best_epoch is not None:
      print('Evaluating epoch', best_epoch)
    else:
      print('Evaluating final epoch')
    if eval_path is not None:
      print('Writing to', eval_path)
    print('Evaluating ckpt', ckpt, 'with eval path', eval_path)
    if test_dataset is not None:
      print('Evaluating on test set')
      test_losses = eval_on_valset(
          test_dataset, agent, FLAGS.regression_loss, FLAGS.l2_weight)
      if summary_writer is not None:
        first_in_test = FLAGS.test_set_start or FLAGS.max_demos_to_load
        last_in_test = first_in_test + FLAGS.test_set_size
        try:
          step = (
              FLAGS.num_epochs if ckpt == 'ckpt'
              else int(ckpt.replace('ckpt_best_of_', '')))
          with summary_writer.as_default():
            for k, v in test_losses.items():
              tf.summary.scalar(
                  f'test_{k}_{first_in_test}-{last_in_test}', v, step=step)
        except ValueError:
          # Not all ckpts will have format ckpt_best_of_{step}.
          pass
        summary_writer.flush()
      with gfile.GFile(eval_path + '_test_loss.pkl', 'wb') as f:
        pickle.dump(test_losses, f)
    del test_dataset
    success_rates = eval_loop.eval_policy(
        env, FLAGS.eval_seed, FLAGS.increment_eval_seed, agent,
        num_eval_episodes, eval_path, num_videos_to_save, summary_writer,
        summary_key=summary_key, stop_if_stuck=FLAGS.stop_if_stuck)
    epoch_to_success_rates[best_epoch] = success_rates


def get_checkpoints_to_evaluate(ckpt_dir, epochs_to_eval=(), exact_ckpts=()):
  """Get file names of all checkpoints to evaluate.

  Args:
    ckpt_dir: Directory from which to load checkpoints.
    epochs_to_eval: Additional epochs to evaluate in addition to final model.
    exact_ckpts: If set, additional exact checkpoint names to evaluate.

  Returns:
    The list of checkpoints to evaluate.
  """
  # None = Current network weights (not necessarily best validation loss).
  ckpts_to_eval = [None]
  if ckpt_dir is not None:  # Evaluate saved checkpoints.
    # ckpts_to_eval = ['ckpt_100']  # Final checkpoint.
    ckpts_to_eval = ['ckpt']  # Final checkpoint.
    for epoch in epochs_to_eval:
      ckpts_to_eval.append(f'ckpt_best_of_{epoch}')
    for ckpt in exact_ckpts:
      if ckpt not in ckpts_to_eval:
        ckpts_to_eval.append(ckpt)
  return ckpts_to_eval
