# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import math
import os
import time

from absl import flags
from absl import logging

from easydict import EasyDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow.compat.v2 as tf
import yaml

from tcc.config import CONFIG

FLAGS = flags.FLAGS


def visualize_batch(data, global_step, batch_size, num_steps):
  """Visualizes a batch."""
  frames = data['frames']
  frames_list = tf.unstack(frames, num=num_steps, axis=1)
  frames_summaries = tf.concat(frames_list, axis=2)
  batch_list = tf.split(frames_summaries, batch_size, axis=0)
  batch_summaries = tf.concat(batch_list, axis=1)
  tf.summary.image('train_batch', batch_summaries, step=global_step)


def visualize_nearest_neighbours(model, data, global_step, batch_size,
                                 num_steps, num_frames_per_step, split):
  """Visualize nearest neighbours in embedding space."""
  # Set learning_phase to False to use models in inference mode.
  tf.keras.backend.set_learning_phase(0)

  cnn = model['cnn']
  emb = model['emb']

  cnn_feats = get_cnn_feats(cnn, data, training=False)
  emb_feats = emb(cnn_feats, num_steps)
  emb_feats = tf.stack(tf.split(emb_feats, num_steps, axis=0), axis=1)

  query_feats = emb_feats[0]

  frames = data['frames']
  image_list = tf.unstack(frames, num=batch_size, axis=0)
  im_list = [image_list[0][num_frames_per_step-1::num_frames_per_step]]
  sim_matrix = np.zeros((batch_size-1, num_steps, num_steps), dtype=np.float32)

  for i in range(1, batch_size):
    candidate_feats = emb_feats[i]

    img_list = tf.unstack(image_list[i], num=num_steps * num_frames_per_step,
                          axis=0)[num_frames_per_step-1::num_frames_per_step]
    nn_img_list = []

    for j in range(num_steps):
      curr_query_feats = tf.tile(query_feats[j:j+1], [num_steps, 1])
      mean_squared_distance = tf.reduce_mean(
          tf.math.squared_difference(curr_query_feats, candidate_feats), axis=1)
      sim_matrix[i-1, j] = softmax(-1.0 * mean_squared_distance)
      nn_img_list.append(img_list[tf.argmin(mean_squared_distance)])

    nn_img = tf.stack(nn_img_list, axis=0)
    im_list.append(nn_img)

  def vstack(im):
    return  tf.concat(tf.unstack(im, num=num_steps), axis=1)

  summary_im = tf.expand_dims(tf.concat([vstack(im) for im in im_list],
                                        axis=0), axis=0)
  tf.summary.image('%s/nn' % split, summary_im, step=global_step)
  # Convert sim_matrix to float32 as summary_image doesn't take float64
  sim_matrix = sim_matrix.astype(np.float32)
  tf.summary.image('%s/similarity_matrix' % split,
                   np.expand_dims(sim_matrix, axis=3), step=global_step)


def softmax(w, t=1.0):
  e = np.exp(np.array(w) / t)
  dist = e / np.sum(e)
  return dist


def random_choice_noreplace(m, n, axis=-1):
  # Generate m random permuations of range (0, n)
  # NumPy version: np.random.rand(m,n).argsort(axis=axis)
  return tf.cast(tf.argsort(tf.random.uniform((m, n)), axis=axis), tf.int64)


def gen_cycles(num_cycles, batch_size, cycle_len):
  """Generate cycles for alignment."""
  random_cycles = random_choice_noreplace(num_cycles, batch_size)[:, :cycle_len]
  return random_cycles


def get_warmup_lr(lr, global_step, lr_params):
  """Returns learning rate during warm up phase."""
  if lr_params.NUM_WARMUP_STEPS > 0:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(lr_params.NUM_WARMUP_STEPS, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_lr = lr_params.INITIAL_LR * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    lr = (1.0 - is_warmup) * lr + is_warmup * warmup_lr
  return lr


# Minimally adapted from Tensorflow object_detection code.
def manual_stepping(global_step, boundaries, rates):
  boundaries = [0] + boundaries
  num_boundaries = len(boundaries)
  rate_index = tf.reduce_max(
      tf.where(
          tf.greater_equal(global_step, boundaries),
          list(range(num_boundaries)), [0] * num_boundaries))
  return tf.reduce_sum(rates * tf.one_hot(rate_index, depth=num_boundaries))


def get_lr_fn(optimizer_config):
  """Returns function that provides current learning rate based on config.

  NOTE: This returns a function as in Eager we need to call assign to update
  the learning rate.

  Args:
    optimizer_config: EasyDict, contains params required to initialize the
      learning rate and the learning rate decay function.
  Returns:
    lr_fn: function, this can be called to return the current learning rate
      based on the provided config.
  Raises:
    ValueError: in case invalid params have been passed in the config.
  """
  lr_params = optimizer_config.LR
  # pylint: disable=g-long-lambda
  if lr_params.DECAY_TYPE == 'exp_decay':
    lr_fn = lambda lr, global_step: tf.train.exponential_decay(
        lr,
        global_step,
        lr_params.EXP_DECAY_STEPS,
        lr_params.EXP_DECAY_RATE,
        staircase=True)()
  elif lr_params.DECAY_TYPE == 'manual':
    lr_step_boundaries = [int(x) for x in lr_params.MANUAL_LR_STEP_BOUNDARIES]

    f = lr_params.MANUAL_LR_DECAY_RATE
    learning_rate_sequence = [(lr_params.INITIAL_LR) * f**p
                              for p in range(len(lr_step_boundaries) + 1)]
    lr_fn = lambda lr, global_step: manual_stepping(
        global_step, lr_step_boundaries, learning_rate_sequence)
  elif lr_params.DECAY_TYPE == 'fixed':
    lr_fn = lambda lr, global_step: lr_params.INITIAL_LR
  elif lr_params.DECAY_TYPE == 'poly':
    lr_fn = lambda lr, global_step: tf.train.polynomial_decay(
        lr,
        global_step,
        CONFIG.TRAIN.MAX_ITERS,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)
  else:
    raise ValueError('Learning rate decay type %s not supported. Only support'
                     'the following decay types: fixed, exp_decay, manual,'
                     'and poly.')

  return (lambda lr, global_step: get_warmup_lr(lr_fn(lr, global_step),
                                                global_step, lr_params))


def get_optimizer(optimizer_config, learning_rate):
  """Returns optimizer based on config and learning rate."""
  if optimizer_config.TYPE == 'AdamOptimizer':
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif optimizer_config.TYPE == 'MomentumOptimizer':
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
  else:
    raise ValueError('Optimizer %s not supported. Only support the following'
                     'optimizers: AdamOptimizer, MomentumOptimizer .')
  return opt


def get_lr_opt_global_step():
  """Intializes learning rate, optimizer and global step."""
  optimizer = get_optimizer(CONFIG.OPTIMIZER, CONFIG.OPTIMIZER.LR.INITIAL_LR)
  global_step = optimizer.iterations
  learning_rate = optimizer.learning_rate
  return learning_rate, optimizer, global_step


def restore_ckpt(logdir, **ckpt_objects):
  """Create and restore checkpoint (if one exists on the path)."""
  # Instantiate checkpoint and restore from any pre-existing checkpoint.
  # Since model is a dict we can insert multiple modular networks in this dict.
  checkpoint = tf.train.Checkpoint(**ckpt_objects)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=logdir,
      max_to_keep=10,
      keep_checkpoint_every_n_hours=1)
  status = checkpoint.restore(ckpt_manager.latest_checkpoint)
  return ckpt_manager, status, checkpoint


def to_dict(config):
  if isinstance(config, list):
    return [to_dict(c) for c in config]
  elif isinstance(config, EasyDict):
    return dict([(k, to_dict(v)) for k, v in config.items()])
  else:
    return config


def setup_train_dir(logdir):
  """Setups directory for training."""
  tf.io.gfile.makedirs(logdir)
  config_path = os.path.join(logdir, 'config.yml')
  if not os.path.exists(config_path):
    logging.info(
        'Using config from config.py as no config.yml file exists in '
        '%s', logdir)
    with  tf.io.gfile.GFile(config_path, 'w') as config_file:
      config = dict([(k, to_dict(v)) for k, v in CONFIG.items()])
      yaml.safe_dump(config, config_file, default_flow_style=False)
  else:
    logging.info('Using config from config.yml that exists in %s.', logdir)
    with tf.io.gfile.GFile(config_path, 'r') as config_file:
      config_dict = yaml.safe_load(config_file)
    CONFIG.update(config_dict)

  train_logs_dir = os.path.join(logdir, 'train_logs')
  if os.path.exists(train_logs_dir) and not FLAGS.force_train:
    raise ValueError('You might be overwriting a directory that already '
                     'has train_logs. Please provide a new logdir name in '
                     'config or pass --force_train while launching script.')
  tf.io.gfile.makedirs(train_logs_dir)


def setup_eval_dir(logdir, config_timeout_seconds=1):
  """Setups directory for evaluation."""
  tf.io.gfile.makedirs(logdir)
  tf.io.gfile.makedirs(os.path.join(logdir, 'eval_logs'))
  config_path = os.path.join(logdir, 'config.yml')
  while not tf.io.gfile.exists(config_path):
    logging.info('Waiting for config to exist. Going to sleep '
                 ' %s for  secs.', config_timeout_seconds)
    time.sleep(config_timeout_seconds)

  while True:
    with tf.io.gfile.GFile(config_path, 'r') as config_file:
      config_dict = yaml.safe_load(config_file)
    if config_dict is None:
      time.sleep(config_timeout_seconds)
    else:
      break
  CONFIG.update(config_dict)


def get_data(iterator):
  """Return a data dict which contains all the requested sequences."""
  data = iterator.get_next()
  return data, data['chosen_steps'], data['seq_lens']


def get_cnn_feats(cnn, data, training, num_steps=None):
  """Passes data through base CNN."""
  if num_steps is None:
    if training:
      num_steps = CONFIG.TRAIN.NUM_FRAMES * CONFIG.DATA.NUM_STEPS
    else:
      num_steps = CONFIG.EVAL.NUM_FRAMES * CONFIG.DATA.NUM_STEPS

  cnn.num_steps = num_steps
  cnn_feats = cnn(data['frames'])
  return cnn_feats


def get_context_steps(step):
  num_steps = CONFIG.DATA.NUM_STEPS
  stride = CONFIG.DATA.FRAME_STRIDE
  # We don't want to see the future.
  steps = np.arange(step - (num_steps - 1) * stride, step + stride, stride)
  return steps


def get_indices(curr_idx, num_steps, seq_len):
  steps = range(curr_idx, curr_idx + num_steps)
  single_steps = np.concatenate([get_context_steps(step) for step in steps])
  single_steps = np.maximum(0, single_steps)
  single_steps = np.minimum(seq_len, single_steps)
  return single_steps


# TODO(debidatta): Modular and simpler function for embedding datasets
# with different embedders.
def get_embeddings_dataset(model, iterator, frames_per_batch,
                           keep_data=False, keep_labels=True,
                           max_embs=None):
  """Get embeddings from a one epoch iterator."""
  keep_labels = keep_labels and CONFIG.DATA.FRAME_LABELS
  num_frames_per_step = CONFIG.DATA.NUM_STEPS
  cnn = model['cnn']
  emb = model['emb']
  embs_list = []
  labels_list = []
  steps_list = []
  seq_lens_list = []
  names_list = []
  seq_labels_list = []
  if keep_data:
    frames_list = []

  n = 0
  def cond(n):
    if max_embs is None:
      return True
    else:
      return n < max_embs

  # Make Recurrent Layers stateful, set batch size.
  # We do this as we are embedding the whole sequence and that can take
  # more than one batch to be passed and we don't want to automatically
  # reset hidden states after each batch.
  if CONFIG.MODEL.EMBEDDER_TYPE == 'convgru':
    for gru_layer in emb.gru_layers:
      gru_layer.stateful = True
      gru_layer.input_spec[0].shape = [1,]

  while cond(n):
    try:
      embs = []
      labels = []
      steps = []
      seq_lens = []
      names = []
      seq_labels = []
      if keep_data:
        frames = []

      # Reset GRU states for each video.
      if CONFIG.MODEL.EMBEDDER_TYPE == 'convgru':
        for gru_layer in emb.gru_layers:
          gru_layer.reset_states()

      data, chosen_steps, seq_len = get_data(iterator)
      seq_len = seq_len.numpy()[0]
      num_batches = int(math.ceil(float(seq_len)/frames_per_batch))
      for i in range(num_batches):
        if  (i + 1) * frames_per_batch > seq_len:
          num_steps = seq_len - i * frames_per_batch
        else:
          num_steps = frames_per_batch
        curr_idx = i * frames_per_batch

        curr_data = {}
        for k, v in data.items():
          # Need to do this as some modalities might not exist.
          if len(v.shape) > 1 and v.shape[1] != 0:
            idxes = get_indices(curr_idx, num_steps, seq_len)
            curr_data[k] = tf.gather(v, idxes, axis=1)
          else:
            curr_data[k] = v

        cnn_feats = get_cnn_feats(cnn, curr_data,
                                  num_steps=num_frames_per_step * num_steps,
                                  training=False)

        emb_feats = emb(cnn_feats, num_steps)
        logging.info('On sequence number %d, frames embedded %d', n,
                     curr_idx + num_steps)
        embs.append(emb_feats.numpy())

      steps.append(chosen_steps.numpy()[0])
      seq_lens.append(seq_len * [seq_len])
      all_labels = data['frame_labels'].numpy()[0]
      name = data['name'].numpy()[0]
      names.append(seq_len * [name])
      seq_label = data['seq_labels'].numpy()[0]
      seq_labels.append(seq_len * [seq_label])
      labels.append(all_labels)
      embs = np.concatenate(embs, axis=0)
      labels = np.concatenate(labels, axis=0)

      steps = np.concatenate(steps, axis=0)
      seq_lens = np.concatenate(seq_lens, axis=0)
      names = np.concatenate(names, axis=0)
      seq_labels = np.concatenate(seq_labels, axis=0)
      if keep_data:
        frames.append(data['frames'].numpy()[0])
        frames = np.concatenate(frames, axis=0)

      if keep_labels:
        labels = labels[~np.isnan(embs).any(axis=1)]
        assert len(embs) == len(labels)
      seq_labels = seq_labels[~np.isnan(embs).any(axis=1)]

      names = names[~np.isnan(embs).any(axis=1)]
      seq_lens = seq_lens[~np.isnan(embs).any(axis=1)]
      steps = steps[~np.isnan(embs).any(axis=1)]
      if keep_data:
        frames = frames[~np.isnan(embs).any(axis=1)]
      embs = embs[~np.isnan(embs).any(axis=1)]

      assert len(embs) == len(seq_lens)
      assert len(embs) == len(steps)
      assert len(names) == len(steps)

      embs_list.append(embs)
      if keep_labels:
        labels_list.append(labels)
      seq_labels_list.append(seq_labels)
      steps_list.append(steps)
      seq_lens_list.append(seq_lens)
      names_list.append(names)
      if keep_data:
        frames_list.append(frames)
      n += 1
    except tf.errors.OutOfRangeError:
      logging.info('Finished embedding the dataset.')
      break

  dataset = {'embs': embs_list,
             'seq_lens': seq_lens_list,
             'steps': steps_list,
             'names': names_list,
             'seq_labels': seq_labels_list}
  if keep_data:
    dataset['frames'] = frames_list
  if keep_labels:
    dataset['labels'] = labels_list

  # Reset statefulness to recurrent layers for other evaluation tasks.
  if CONFIG.MODEL.EMBEDDER_TYPE == 'convgru':
    for gru_layer in emb.gru_layers:
      gru_layer.stateful = False

  return dataset


def gen_plot(x, y):
  """Create a pyplot, save to buffer and return TB compatible image."""
  plt.figure()
  plt.plot(x, y)
  plt.title('Val Accuracy')
  plt.ylim(0, 1)
  plt.tight_layout()
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


class Stopwatch(object):
  """Simple timer for measuring elapsed time."""

  def __init__(self):
    self.reset()

  def elapsed(self):
    return time.time() - self.time

  def done(self, target_interval):
    return self.elapsed() >= target_interval

  def reset(self):
    self.time = time.time()


def set_learning_phase(f):
  """Sets the correct learning phase before calling function f."""
  def wrapper(*args, **kwargs):
    """Calls the function f after setting proper learning phase."""
    if 'training' not in kwargs:
      raise ValueError('Function called with set_learning_phase decorator which'
                       ' does not have training argument.')
    training = kwargs['training']
    if training:
      # Set learning_phase to True to use models in training mode.
      tf.keras.backend.set_learning_phase(1)
    else:
      # Set learning_phase to False to use models in inference mode.
      tf.keras.backend.set_learning_phase(0)
    return f(*args, **kwargs)
  return wrapper
