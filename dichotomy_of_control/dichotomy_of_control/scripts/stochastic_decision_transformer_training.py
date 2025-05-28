# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tensorflow trainer for stochasticDecisionTransformers."""
import time
import numpy as np
import tensorflow as tf
from dichotomy_of_control import utils


def discount_cumsum(x, gamma):
  """Cumulative sum a vector with geometric discounting.

  Args:
    x: A single dimensional np.ndarray.
    gamma: A float between 0 and 1 for discounting.

  Returns:
    discounted_x: The resulting np.ndarray.
  """
  if x.size == 0:
    return np.array([[] for _ in range(x.shape[0])])
  discounted_x = np.zeros_like(x)
  discounted_x[-1] = x[-1]
  for t in reversed(range(x.shape[0] - 1)):
    discounted_x[t] = x[t] + gamma * discounted_x[t + 1]
  return discounted_x


class StochasticSequenceDataLoader:
  """Stores trajectory data and provides interface for sampling batches.

  Based on get_batch function in decision transformer code.

  Reference: https://github.com/kzl/decision-transformer

  TODO(b/194820797): refactor to use more standard tensorflow patterns
    and avoid redundant computation within minibatch sampling.
  """

  def __init__(self,
               trajectories,
               context_len,
               max_ep_len,
               batch_size,
               state_mean,
               state_std,
               scale,
               future_len=None,
               gamma=1.):
    """Initializes a SequenceDataLoader from trajectories.

    Args:
      trajectories: A list of dicts with keys 'observations', 'actions',
        'rewards', and 'dones'. Each dict represents one trajectory. All entries
        are numpy arrays.
      context_len: An int context length.
      max_ep_len: An int maximum episode length.
      batch_size: An int batch size.
      state_mean: A np.ndarray for normalizing states.
      state_std: A np.ndarray for normalizing states.
      scale: A float scaling factor for returns.
    """
    if not trajectories:
      raise ValueError('SequenceDataLoader must have at least one trajectory.')
    self._future_len = future_len or context_len
    self._trajectories = trajectories
    self._context_len = context_len
    self._max_ep_len = max_ep_len
    self._batch_size = batch_size
    self._state_mean = state_mean
    self._state_std = state_std
    self._scale = scale
    self._gamma = gamma
    self._num_trajectories = len(trajectories)

    # get trajectory lengths and states and action dimensions
    traj_lens = []
    self._state_dim = None
    self._act_dim = None
    for path in trajectories:
      traj_lens.append(len(path['observations']))
      if self._state_dim is None:
        assert len(path['observations'].shape) == 2
        self._state_dim = path['observations'].shape[1]
      if self._act_dim is None:
        assert len(path['actions'].shape) == 2
        self._act_dim = path['actions'].shape[1]
      assert self._state_dim == path['observations'].shape[1]
      assert self._act_dim == path['actions'].shape[1]
    traj_lens = np.array(traj_lens)

    # used to reweight sampling so we sample according to timesteps
    self._p_sample = traj_lens / sum(traj_lens)

  def get_batch(self):
    """Sample a batch of trajectories as tf.Tensors.

    Returns:
      s: A tf.Tensor of batches of state trajectories of shape `[B, T, ...]`.
      a: A tf.Tensor of batches of action trajectories of shape `[B, T, ...]`.
      r: A tf.Tensor of batches of reward trajectories of shape `[B, T]`.
      d: A tf.Tensor of batches of done trajectories of shape `[B, T]`.
      rtg: A tf.Tensor of batches of returns-to-go of shape `[B, T]`.
      timesteps: A tf.Tensor of batches of timesteps of shape `[B, T, ...]`.
      mask: A tf.Tensor of batches of boolean masks over which indices can be
        attended to, of shape `[B, T]`.
    """
    batch_inds = np.random.choice(
        np.arange(self._num_trajectories),
        size=self._batch_size,
        replace=True,
        p=self._p_sample,  # reweights so we sample according to timesteps
    )
    max_len = self._context_len
    future_max_len = self._future_len

    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    fs, fa, fr, frtg, ftimesteps, fmask = [], [], [], [], [], []
    for i in range(self._batch_size):
      traj = self._trajectories[int(batch_inds[i])]
      si = np.random.randint(0, traj['rewards'].shape[0])

      # get sequences from dataset
      s.append(traj['observations'][si:si + max_len].reshape(
          [1, -1, self._state_dim]))
      a.append(traj['actions'][si:si + max_len].reshape(1, -1, self._act_dim))
      r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
      if 'terminals' in traj:
        d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
      else:
        d.append(traj['dones'][si:si + max_len].reshape(1, -1))
      timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
      timesteps[-1][timesteps[-1] >= self._max_ep_len] = self._max_ep_len - 1
      rtg.append(
          discount_cumsum(traj['rewards'][si:],
                          gamma=self._gamma)[:s[-1].shape[1] + 1].reshape(
                              1, -1, 1))
      if rtg[-1].shape[1] <= s[-1].shape[1]:
        rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

      # get future sequenes from dataset
      fs.append(traj['observations'][si + max_len:si + max_len +
                                     future_max_len].reshape(
                                         [1, -1, self._state_dim]))
      fa.append(traj['actions'][si + max_len:si + max_len +
                                future_max_len].reshape(1, -1, self._act_dim))
      fr.append(traj['rewards'][si + max_len:si + max_len +
                                future_max_len].reshape(1, -1, 1))
      ftimesteps.append(
          np.arange(si + max_len,
                    si + max_len + fs[-1].shape[1]).reshape(1, -1))
      ftimesteps[-1][ftimesteps[-1] >= self._max_ep_len] = self._max_ep_len - 1
      frtg.append(
          discount_cumsum(traj['rewards'][si + max_len:],
                          gamma=self._gamma)[:fs[-1].shape[1] + 1].reshape(
                              1, -1, 1))
      if frtg[-1].shape[1] <= fs[-1].shape[1]:
        frtg[-1] = np.concatenate([frtg[-1], np.zeros((1, 1, 1))], axis=1)

      # padding and state + reward normalization
      tlen = s[-1].shape[1]
      s[-1] = np.concatenate(
          [np.zeros([1, max_len - tlen, self._state_dim]), s[-1]], axis=1)
      s[-1] = (s[-1] - self._state_mean) / self._state_std
      a[-1] = np.concatenate(
          [np.ones((1, max_len - tlen, self._act_dim)) * -10., a[-1]], axis=1)
      r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
      d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
      rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]],
                               axis=1) / self._scale
      timesteps[-1] = np.concatenate(
          [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
      mask.append(
          np.concatenate([np.zeros((1, max_len - tlen)),
                          np.ones((1, tlen))],
                         axis=1))
      ftlen = fs[-1].shape[1]
      fs[-1] = np.concatenate(
          [np.zeros([1, future_max_len - ftlen, self._state_dim]), fs[-1]],
          axis=1)
      fs[-1] = (fs[-1] - self._state_mean) / self._state_std
      fa[-1] = np.concatenate(
          [np.ones((1, future_max_len - ftlen, self._act_dim)) * -10., fa[-1]],
          axis=1)
      fr[-1] = np.concatenate(
          [np.zeros((1, future_max_len - ftlen, 1)), fr[-1]], axis=1)
      frtg[-1] = np.concatenate(
          [np.zeros(
              (1, future_max_len - ftlen, 1)), frtg[-1]], axis=1) / self._scale
      ftimesteps[-1] = np.concatenate(
          [np.zeros((1, future_max_len - ftlen)), ftimesteps[-1]], axis=1)
      fmask.append(
          np.concatenate(
              [np.zeros((1, future_max_len - ftlen)),
               np.ones((1, ftlen))],
              axis=1))

    s = tf.convert_to_tensor(np.concatenate(s, axis=0), dtype=tf.float32)
    a = tf.convert_to_tensor(np.concatenate(a, axis=0), dtype=tf.float32)
    r = tf.convert_to_tensor(np.concatenate(r, axis=0), dtype=tf.float32)
    d = tf.convert_to_tensor(np.concatenate(d, axis=0), dtype=tf.float32)
    rtg = tf.convert_to_tensor(np.concatenate(rtg, axis=0), dtype=tf.float32)
    timesteps = tf.convert_to_tensor(
        np.concatenate(timesteps, axis=0), dtype=tf.int64)
    mask = tf.convert_to_tensor(np.concatenate(mask, axis=0), dtype=tf.float32)
    fs = tf.convert_to_tensor(np.concatenate(fs, axis=0), dtype=tf.float32)
    fa = tf.convert_to_tensor(np.concatenate(fa, axis=0), dtype=tf.float32)
    fr = tf.convert_to_tensor(np.concatenate(fr, axis=0), dtype=tf.float32)
    frtg = tf.convert_to_tensor(np.concatenate(frtg, axis=0), dtype=tf.float32)
    ftimesteps = tf.convert_to_tensor(
        np.concatenate(ftimesteps, axis=0), dtype=tf.int64)
    fmask = tf.convert_to_tensor(
        np.concatenate(fmask, axis=0), dtype=tf.float32)

    return s, a, r, d, rtg, timesteps, mask, fs, fa, fr, frtg, ftimesteps, fmask


class StochasticDecisionTransformerTrainer:
  """Tensorflow trainer for DecisionTransformers.

  Ported from PyTorch SequenceTrainer class in decision transformer code.

  Reference: https://github.com/kzl/decision-transformer
  """

  def __init__(self,
               model,
               optimizer,
               data_loader,
               loss_fn,
               eval_fns=None,
               prior_weight=1.0,
               energy_weight=1.0):
    """Initializes a DecisionTransformerTrainer.

    Args:
      model: A DecisionTransformer model.
      optimizer: A tf.keras.Optimizer.
      data_loader: A SequenceDataLoader holding trajectory data.
      loss_fn: A tf loss function.
      eval_fns: A list of functions of the model to run after each train iter.
    """
    self._model = model
    self._optimizer = optimizer
    self._data_loader = data_loader
    self._loss_fn = loss_fn
    self._eval_fns = [] if eval_fns is None else eval_fns
    self._prior_weight = prior_weight
    self._energy_weight = energy_weight

    self._start_time = time.time()

  def train_iteration(self, num_steps, iter_num=0, print_logs=False):
    """Run a single training epoch.

    Args: num_steps : Int number of batches to sample within epoch.
      iter_num: Int for logging.
      print_logs: Bool indicating whether to print logs or not.

    Returns:
      logs : dict of str to Any with log information.
    """
    train_losses = []
    logs = dict()

    train_start = time.time()

    for _ in range(num_steps):
      train_loss, train_info = self.train_step()
      train_losses.append(train_loss)

    logs['time/training'] = time.time() - train_start

    eval_start = time.time()

    for eval_fn in self._eval_fns:
      outputs = eval_fn(self._model)
      for k, v in outputs.items():
        logs[f'evaluation/{k}'] = v

    logs['time/total'] = time.time() - self._start_time
    logs['time/evaluation'] = time.time() - eval_start
    logs['training/train_loss_mean'] = np.mean(train_losses)
    logs['training/train_loss_std'] = np.std(train_losses)
    logs = {**logs, **train_info}

    if print_logs:
      print('=' * 80)
      print(f'Iteration {iter_num}')
      for k, v in logs.items():
        print(f'{k}: {v}')
      print(flush=True)

    return logs

  def train_step(self):
    """Run a single training step with one batch of data.

    Returns:
      loss: float loss.
    """
    batch = self._data_loader.get_batch()
    (states, actions, rewards, _, rtg, timesteps, attention_mask, future_states,
     future_actions, future_rewards, future_rtg, future_timesteps,
     future_attention_mask) = batch
    action_target = tf.identity(actions)
    value_target = rtg[:, :-1]

    with tf.GradientTape() as tape:
      (action_preds, value_preds, f_pred, f_mean, f_logvar, prior_mean,
       prior_logvar, energies) = self._model(
           states,
           actions,
           rewards,
           timesteps,
           attention_mask,
           future_states,
           future_actions,
           future_rewards,
           future_timesteps,
           future_attention_mask,
           training=True,
       )
      act_dim = action_preds.shape[2]
      predicted_idx_mask = (tf.reshape(attention_mask, (-1,)) > 0)
      action_preds = tf.reshape(action_preds, (-1, act_dim))
      action_target = tf.reshape(action_target, (-1, act_dim))
      action_preds = tf.boolean_mask(action_preds, predicted_idx_mask)
      action_target = tf.boolean_mask(action_target, predicted_idx_mask)

      value_preds = tf.reshape(value_preds, (-1))
      value_target = tf.reshape(value_target, (-1))
      value_preds = tf.boolean_mask(value_preds, predicted_idx_mask)
      value_target = tf.boolean_mask(value_target, predicted_idx_mask)

      action_loss = self._loss_fn(action_preds, action_target)
      action_acc = tf.reduce_sum(
          tf.cast(
              tf.argmax(action_preds, axis=-1) == tf.argmax(
                  action_target, axis=-1), tf.float32)) / tf.reduce_sum(
                      tf.cast(predicted_idx_mask, tf.float32))
      value_loss = self._loss_fn(value_preds, value_target)
      if self._energy_weight:
        prior_mean = tf.stop_gradient(prior_mean)
        prior_logvar = tf.stop_gradient(prior_logvar)
      prior_loss = tf.reduce_mean(
          utils.dense_gaussian_kl(f_mean, f_logvar, prior_mean, prior_logvar))

      pos_loss = tf.linalg.diag_part(energies)
      neg_loss = tf.reduce_logsumexp(energies, axis=-1)
      energy_loss = tf.reduce_mean(-pos_loss + neg_loss)

      loss = action_loss + value_loss + self._prior_weight * prior_loss + self._energy_weight * energy_loss
      info = {
          'action_loss': action_loss,
          'value_loss': value_loss,
          'value_loss': value_loss,
          'energy_loss': energy_loss,
          'prior_loss': prior_loss,
          'loss': loss,
          'action_acc': action_acc
      }

    grads = tape.gradient(loss, self._model.trainable_weights)
    self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
    return float(loss), info
