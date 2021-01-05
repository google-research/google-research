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

"""Training and eval worker utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

from . import logging_utils

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf


class BaseModel(object):

  def train_fn(self, x_bhwc):
    raise NotImplementedError

  def eval_fn(self, x_bhwc):
    raise NotImplementedError

  def samples_fn(self, x_bhwc):
    raise NotImplementedError

  @property
  def trainable_variables(self):
    raise NotImplementedError

  @property
  def ema(self):
    raise NotImplementedError


def _make_ema_model(orig_model, model_constructor):

  # Model with EMA parameters
  if orig_model.ema is None:
    return None

  def _to_original_variable_name(name):
    # map to the original variable name
    parts = name.split('/')
    assert parts[0] == 'ema_scope'
    return '/'.join(parts[1:])

  def _ema_getter(getter, name, *args, **kwargs):
    v = getter(_to_original_variable_name(name), *args, **kwargs)
    v = orig_model.ema.average(v)
    if v is None:
      raise RuntimeError('invalid EMA variable name {} -> {}'.format(
          name, _to_original_variable_name(name)))
    return v

  with tf.variable_scope(
      tf.get_variable_scope(), custom_getter=_ema_getter, reuse=True):
    with tf.name_scope('ema_scope'):
      return model_constructor()


def run_eval(
    model_constructor,
    logdir,
    total_bs,
    master,
    input_fn,
    dataset_size):

  worker = EvalWorker(
      master=master,
      model_constructor=model_constructor,
      total_bs=total_bs,
      input_fn=input_fn)
  worker.run(logdir=logdir, once=True)


class EvalWorker(object):

  def __init__(self, master, model_constructor, total_bs, input_fn):
    self.strategy = tf.distribute.MirroredStrategy()

    self.num_cores = self.strategy.num_replicas_in_sync
    assert total_bs % self.num_cores == 0
    self.total_bs = total_bs
    self.local_bs = total_bs // self.num_cores
    logging.info('num cores: {}'.format(self.num_cores))
    logging.info('total batch size: {}'.format(self.total_bs))
    logging.info('local batch size: {}'.format(self.local_bs))

    with self.strategy.scope():
      # Dataset iterator
      dataset = input_fn(params={'batch_size': self.total_bs})
      self.eval_iterator = self.strategy.experimental_distribute_dataset(
          dataset).make_initializable_iterator()
      eval_iterator_next = next(self.eval_iterator)

      # Model
      self.model = model_constructor()
      # Model with EMA parameters
      self.ema_model = _make_ema_model(self.model, model_constructor)

      # Global step
      self.global_step = tf.train.get_global_step()
      assert self.global_step is not None, 'global step not created'

      # Eval/samples graphs
      self.eval_outputs = self._distributed(
          self.model.eval_fn, args=(eval_iterator_next,), reduction='mean')
      self.samples_outputs = self._distributed(
          self.model.samples_fn, args=(eval_iterator_next,), reduction='concat')
      # EMA versions of the above
      if self.ema_model is not None:
        self.ema_eval_outputs = self._distributed(
            self.ema_model.eval_fn,
            args=(eval_iterator_next,),
            reduction='mean')
        self.ema_samples_outputs = self._distributed(
            self.ema_model.samples_fn,
            args=(eval_iterator_next,),
            reduction='concat')

  def _distributed(self, model_fn, args, reduction):
    """Sharded computation."""

    def model_wrapper(inputs_):
      return model_fn(inputs_['image'])

    out = self.strategy.run(model_wrapper, args=args)
    assert isinstance(out, dict)

    if reduction == 'mean':
      out = {
          k: tf.reduce_mean(self.strategy.reduce('mean', v))
          for k, v in out.items()
      }
      assert all(v.shape == [] for v in out.values())  # pylint: disable=g-explicit-bool-comparison
    elif reduction == 'concat':
      out = {
          k: tf.concat(self.strategy.experimental_local_results(v), axis=0)
          for k, v in out.items()
      }
      assert all(v.shape[0] == self.total_bs for v in out.values())
    else:
      raise NotImplementedError(reduction)

    return out

  def _make_session(self):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    logging.info('making session...')
    return tf.Session(config=config)

  def _run_eval(self, sess, ema):
    logging.info('eval pass...')
    sess.run(self.eval_iterator.initializer)
    all_loss_lists = collections.defaultdict(list)
    run_times = []
    try:
      while True:
        # Log progress
        if run_times and len(run_times) % 100 == 0:
          num_batches_seen = len(list(all_loss_lists.values())[0])
          logging.info(
              'eval examples_so_far={} time_per_batch={:.5f} {}'.format(
                  num_batches_seen * self.total_bs,
                  np.mean(run_times[1:]),
                  {k: np.mean(l) for k, l in all_loss_lists.items()}))
        tstart = time.time()
        results = sess.run(self.ema_eval_outputs if ema else self.eval_outputs)
        run_times.append(time.time() - tstart)
        for k, v in results.items():
          all_loss_lists[k].append(v)
    except tf.errors.OutOfRangeError:
      pass
    num_batches_seen = len(list(all_loss_lists.values())[0])
    logging.info('eval pass done ({} batches, {} examples)'.format(
        num_batches_seen, num_batches_seen * self.total_bs))
    results = {k: np.mean(l) for k, l in all_loss_lists.items()}
    logging.info('final eval results: {}'.format(results))
    return results

  def _run_sampling(self, sess, ema):
    sess.run(self.eval_iterator.initializer)
    logging.info('sampling...')
    samples = sess.run(
        self.ema_samples_outputs if ema else self.samples_outputs)
    logging.info('sampling done')
    return samples

  def _write_eval_and_samples(self, sess, log, curr_step, prefix, ema):
    # Samples
    samples_dict = self._run_sampling(sess, ema=ema)
    for k, v in samples_dict.items():
      assert len(v.shape) == 4 and v.shape[0] == self.total_bs
      log.summary_writer.images(
          '{}/{}'.format(prefix, k),
          np.clip(v, 0, 255).astype('uint8'),
          step=curr_step)
    log.summary_writer.flush()

    # Eval
    eval_losses = self._run_eval(sess, ema=ema)
    for k, v in eval_losses.items():
      log.write(prefix, [{k: v}], step=curr_step)

  def run(self, logdir, once, skip_non_ema_pass=True):
    """Runs the eval/sampling worker loop.

    Args:
      logdir: directory to read checkpoints from
      once: if True, writes results to a temporary directory (not to logdir),
        and exits after evaluating one checkpoint.
    """
    if once:
      eval_logdir = os.path.join(logdir, 'eval_once_{}'.format(time.time()))
    else:
      eval_logdir = logdir
    logging.info('Writing eval data to: {}'.format(eval_logdir))
    eval_log = logging_utils.Log(eval_logdir, write_graph=False)

    with self._make_session() as sess:
      # Checkpoint loading
      logging.info('making saver')
      saver = tf.train.Saver()

      for ckpt in tf.train.checkpoints_iterator(logdir):
        logging.info('restoring params...')
        saver.restore(sess, ckpt)
        global_step_val = sess.run(self.global_step)
        logging.info('restored global step: {}'.format(global_step_val))

        if not skip_non_ema_pass:
          logging.info('non-ema pass')
          self._write_eval_and_samples(
              sess,
              log=eval_log,
              curr_step=global_step_val,
              prefix='eval',
              ema=False)

        if self.ema_model is not None:
          logging.info('ema pass')
          self._write_eval_and_samples(
              sess,
              log=eval_log,
              curr_step=global_step_val,
              prefix='eval_ema',
              ema=True)

        if once:
          break
