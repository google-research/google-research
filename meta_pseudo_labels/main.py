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

# pylint: disable=logging-format-interpolation
# pylint: disable=unused-import
# pylint: disable=g-long-lambda
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=line-too-long

r"""Docs."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import itertools
import json
import os
import sys
import threading
import time
import traceback

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from meta_pseudo_labels import common_utils
from meta_pseudo_labels import data_utils
from meta_pseudo_labels import flag_utils
from meta_pseudo_labels import modeling
from meta_pseudo_labels import training_utils
from tensorflow.python.tpu import tpu_function


FLAGS = flags.FLAGS

MAX_RETRIES = 50

if 'gfile' not in sys.modules:
  gfile = tf.gfile


def set_tpu_info(params):
  """Docs."""
  logging.info('Retrieve TPU information')
  tpu_init = tf.tpu.initialize_system()
  tpu_shutdown = tf.tpu.shutdown_system()
  with common_utils.get_session(params, isolate_session_state=True) as sess:
    topology_proto = sess.run(tpu_init)
    topology = tpu_lib.topology.Topology(serialized=topology_proto)
    sess.run(tpu_shutdown)

  # Note: we use the name `worker` instead of `replica`
  # TPU terminologies are extremely confusing.
  num_cores_per_replica = params.num_cores_per_replica
  num_workers = topology.num_tasks
  num_replicas = (topology.num_tpus_per_task * topology.num_tasks //
                  num_cores_per_replica)
  params.add_hparam('num_workers', num_workers)
  params.add_hparam('num_replicas', num_replicas)
  params.add_hparam('num_cores_per_worker', topology.num_tpus_per_task)

  logging.info('-' * 80)
  logging.info(f'num_workers={num_workers}')
  logging.info(f'num_cores_per_worker={topology.num_tpus_per_task}')
  logging.info(f'num_cores_per_replica={num_cores_per_replica}')
  logging.info(f'num_replicas={num_replicas}')

  params.topology = topology
  num_cores_to_shape = {
      1: [1, 1, 1, 1],
      2: [1, 1, 1, 2],
      4: [1, 2, 1, 2],
      8: [2, 2, 1, 2],
      16: [4, 2, 1, 2],
      32: [4, 4, 1, 2],
      64: [4, 8, 1, 2],
      128: [8, 8, 1, 2],
      256: [8, 16, 1, 2],
      512: [16, 16, 1, 2],
      2048: [32, 32, 1, 2],
  }
  if params.num_cores_per_replica not in num_cores_to_shape:
    raise ValueError(f'Unknown num_cores {params.num_cores_per_replica}')
  computation_shape = num_cores_to_shape[params.num_cores_per_replica]
  params.device_assignment = tpu_lib.device_assignment.device_assignment(
      topology=params.topology,
      computation_shape=computation_shape,
      num_replicas=params.num_replicas)

  if (params.batch_norm_batch_size is not None and
      params.batch_norm_batch_size < params.train_batch_size // num_workers):
    logging.warning(f'batch_norm_batch_size={FLAGS.batch_norm_batch_size} '
                    f'changed into {FLAGS.train_batch_size}')
    params['batch_norm_batch_size'] = params.train_batch_size // num_workers


def get_model_builder(params):
  """Return the function that builds models."""
  if params.model_type == 'wrn-28-2':
    return modeling.Wrn28k(params, k=2)
  elif params.model_type == 'wrn-28-10':
    return modeling.Wrn28k(params, k=10)
  elif params.model_type == 'wrn-28-large':
    return modeling.Wrn28k(params, k=135)
  elif params.model_type == 'resnet-50':
    return modeling.ResNet50(params)
  elif params.model_type.startswith('efficientnet-'):
    model = modeling.EfficientNet(params)
    if 'imagenet' in params.dataset_name.lower():
      params['eval_image_size'] = model.eval_image_size
    return model
  elif params.model_type.startswith('nas-'):
    model = modeling.CustomNet(params)
    if 'imagenet' in params.dataset_name.lower():
      params['eval_image_size'] = model.eval_image_size
    return model
  else:
    raise ValueError(f'Unknown model_type `{params.model_type}`')


def prepare_eval(params, model, eval_logdir, while_training=False):
  """Docs."""

  (eval_infeed_ops,
   eval_infeed_graphs, eval_size) = data_utils.build_eval_infeeds(params)
  eval_infeed_thread = common_utils.InfeedThread(
      params=params,
      infeed_ops=eval_infeed_ops,
      infeed_graphs=eval_infeed_graphs,
      name='eval_infeed')
  num_eval_steps = eval_size // params.eval_batch_size

  logging.info(f'Each eval will run for {num_eval_steps} steps')
  def eval_loop():
    """Docs."""
    def _cond(step, *args):  # pylint: disable=unused-argument
      return tf.less(step, tf.cast(num_eval_steps, step.dtype))
    def _body(step, *args):
      outs = training_utils.eval_step_fn(params, model)
      new_args = [step+1] + [a.write(step, o) for a, o in zip(args, outs)]
      return tuple(new_args)

    batch_size = batch_size = params.eval_batch_size // params.num_replicas
    num_classes = params.num_classes
    logits = tf.TensorArray(dtype=tf.float32,
                            size=num_eval_steps,
                            element_shape=[batch_size, num_classes])
    labels = tf.TensorArray(dtype=tf.int32,
                            size=num_eval_steps,
                            element_shape=[batch_size, 1])
    masks = tf.TensorArray(dtype=tf.float32,
                           size=num_eval_steps,
                           element_shape=[batch_size])
    loop_inps = [0, logits, labels, masks]
    loop_outs = tf.while_loop(_cond, _body, loop_inps, parallel_iterations=1,
                              name='eval')
    return [o.concat() for o in loop_outs[1:]]

  if while_training:
    with tf.variable_scope('ema', reuse=True):
      eval_op = tf.tpu.shard(
          computation=eval_loop,
          num_shards=params.num_replicas,
          device_assignment=params.device_assignment)
  else:
    eval_op = tf.tpu.shard(
        computation=eval_loop,
        num_shards=params.num_replicas,
        device_assignment=params.device_assignment)

  eval_logdir = os.path.join(params.output_dir, 'logs', eval_logdir)
  if gfile.IsDirectory(eval_logdir):
    gfile.DeleteRecursively(eval_logdir)
    gfile.MakeDirs(eval_logdir, mode=0o777)
  summary_writer = tf.summary.FileWriter(eval_logdir)

  def eval_fn(sess, step):
    """Docs."""
    eval_infeed_thread.start()
    logits, labels, mask = sess.run(eval_op)

    num_examples = np.sum(mask)
    sorted_indices = np.argsort(logits, axis=-1)

    def _top_k(k):
      in_top_k = np.any(sorted_indices[:, -k:] == labels, axis=-1)
      total = np.sum(in_top_k.astype(np.float32) * mask)
      return total / num_examples

    top_1, top_5 = _top_k(k=1), _top_k(k=5)
    tb_step = step // 1000 if params.task_mode == 'eval_forever' else step
    summary_writer.add_summary(
        tf.Summary(value=[
            tf.Summary.Value(tag='eval/top_1', simple_value=top_1),
            tf.Summary.Value(tag='eval/top_5', simple_value=top_5),
        ]),
        tb_step)
    summary_writer.flush()
    log_string = ' '.join([
        f'step={step:<8d}',
        f'total={int(num_examples):<6d}',
        f'top_1={top_1:<8.6f}',
        f'top_5={top_5:<8.6f}',
    ])
    logging.info(log_string)

    weak_result = False
    eval_infeed_thread.join()

    return weak_result, top_1

  return eval_fn, summary_writer


def train_tpu(params, should_eval=False):
  """Training routines."""
  set_tpu_info(params)
  train_graph = tf.Graph()

  # train_op
  infeed_ops, infeed_graphs = data_utils.build_train_infeeds(params)

  with train_graph.as_default():
    model = get_model_builder(params)
    if 'mpl' in params.dataset_name.lower():
      train_class = training_utils.MPL()
    elif 'uda' in params.dataset_name.lower():
      train_class = training_utils.UDA()
    else:
      train_class = training_utils.Supervised()
    @tpu_function.on_device_training_loop
    def train_loop():
      """Docs."""
      def _cond(step):
        return tf.less(step, tf.cast(params.save_every, step.dtype))
      def _body(step):
        run_op = train_class.step_fn(params, model)
        with tf.control_dependencies([run_op]):
          return step+1
      loop_inps = [tf.cast(0, tf.int32)]
      loop_outs = tf.while_loop(_cond, _body, loop_inps,
                                parallel_iterations=1, name='train')
      train_op = loop_outs.op
      return train_op
    train_op = tf.tpu.shard(computation=train_loop,
                            num_shards=params.num_replicas,
                            device_assignment=params.device_assignment)
    global_step = tf.train.get_or_create_global_step()
    reset_global_step = tf.assign(global_step, 0)

    num_params = common_utils.count_params()
    logging.info(f'Model has {num_params} params')

    if should_eval:
      eval_fn, eval_summary_writer = prepare_eval(
          params=params,
          model=model,
          eval_logdir=f'eval_{params.image_size}',
          while_training=True)
      best_acc = -1.

    tf.io.write_graph(train_graph, params.output_dir, 'train.pbtxt',
                      as_text=True)

    # outfeed_dequeue_op
    outfeed_signature = train_class.outfeed_signature()
    outfeed_ops, outfeed_graph = common_utils.get_outfeed_ops(
        params, outfeed_signature)

    # saver
    max_to_keep = 1 if should_eval else None
    saver = common_utils.get_saver(max_to_keep=max_to_keep)
    ckpt_dir = os.path.join(params.output_dir, 'ckpt_last')
    async_checkpoint = common_utils.AsyncCheckpoint(
        saver, ckpt_dir, max_to_keep=max_to_keep)

    if params.load_checkpoint is not None:
      reader = tf.train.NewCheckpointReader(params.load_checkpoint)
      var_to_shape_map = reader.get_variable_to_shape_map()
      var_list = {}
      global_variables = tf.global_variables()
      for v in global_variables:
        v_name = common_utils.strip_var_name(v.name)
        if v_name in var_to_shape_map:
          var_list[v_name] = v
        else:
          logging.info(f'NOT FOUND: v.name={v.name:<100} v_name={v_name}')
      logging.info(
          f'Found {len(var_list)} variables (out of {len(global_variables)})'
          f' in {params.load_checkpoint}')
      restore_saver = tf.train.Saver(var_list=var_list)
    else:
      restore_saver = saver

    # actually run
    tpu_init = tf.tpu.initialize_system()
    var_init = tf.global_variables_initializer()
    tpu_shutdown = tf.tpu.shutdown_system()
    with common_utils.get_session(params) as sess:
      logging.info('Initialize TPU system')
      sess.run(tpu_init)

      run_options = tf.RunOptions(
          timeout_in_ms=1000*60*60*10,  # 10 hours
          report_tensor_allocations_upon_oom=True)
      sess.run(var_init, options=run_options)

      latest_checkpoint = common_utils.get_latest_checkpoint(ckpt_dir)
      if latest_checkpoint is not None:
        logging.info(f'Initialize vars from `{latest_checkpoint}`')
        saver.restore(sess, latest_checkpoint)
      elif params.load_checkpoint is not None:
        logging.info(f'Initialize vars from `{params.load_checkpoint}`')
        restore_saver.restore(sess, params.load_checkpoint)
        if params.load_checkpoint_and_restart_global_step:
          logging.info('Reset global_step to 0')
          sess.run(reset_global_step)
      else:
        logging.info('Initialize vars from scratch')

      infeed_thread = common_utils.InfeedThread(
          params=params,
          infeed_ops=infeed_ops,
          infeed_graphs=infeed_graphs,
          name='train_infeed')
      outfeed_thread = common_utils.OutfeedThread(
          params, outfeed_ops, outfeed_graph, outfeed_signature)
      outfeed_thread.start()

      logging.info('Start training')
      while True:
        step = sess.run(global_step)
        if step >= params.num_train_steps:
          break

        infeed_thread.start()
        sess.run(train_op, options=run_options)
        step = sess.run(global_step)
        async_checkpoint.save(sess, step)
        infeed_thread.join()

        if should_eval:
          weak_result, acc = eval_fn(sess, step)
          if weak_result and not params.running_local_dev:
            logging.info('Weak results. Stop training')
            break
          if best_acc < acc:
            best_acc = acc

      logging.info('Wait for [infeed,outfeed,eval,checkpoint]_thread to stop')
      async_checkpoint.join()
      infeed_thread.stop()
      outfeed_thread.join()
      if should_eval:
        eval_summary_writer.close()
        with gfile.GFile(os.path.join(params.output_dir, 'acc'), 'w') as fout:
          fout.write(f'{best_acc:<.10f}')

      logging.info('Shut down TPU system.')
      sess.run(tpu_shutdown)


def eval_tpu(params):
  """Eval routines."""
  set_tpu_info(params)
  tf.reset_default_graph()

  model = get_model_builder(params)
  if 'eval_image_size' in params:
    image_size = max(params.image_size, params.eval_image_size)
  else:
    image_size = params.image_size
  eval_fn, eval_summary_writer = prepare_eval(
      params=params,
      model=model,
      eval_logdir=f'eval_{image_size}_all',
      while_training=False)
  global_step = tf.train.get_or_create_global_step()

  # saver
  saver = common_utils.get_saver(max_to_keep=None, restore_ema=True)
  ckpt_dir = os.path.join(params.output_dir, 'ckpt_last')

  # best checkpoint
  best_acc = -1.
  best_acc_ckpt_name = None
  best_acc_path = os.path.join(params.output_dir, 'ckpt_best')
  best_acc_file = os.path.join(best_acc_path, 'best_acc')
  if not gfile.IsDirectory(best_acc_path):
    gfile.MakeDirs(best_acc_path)

  # actually run
  tpu_init = tf.tpu.initialize_system()
  tpu_shutdown = tf.tpu.shutdown_system()
  with common_utils.get_session(params) as sess:
    logging.info('Initialize TPU system')
    sess.run(tpu_init)

    all_checkpoints = tf.train.get_checkpoint_state(
        ckpt_dir).all_model_checkpoint_paths
    logging.info('Start eval')
    for ckpt_name in all_checkpoints:
      saver.restore(sess, ckpt_name)
      step = sess.run(global_step)
      _, curr_acc = eval_fn(sess, step)
      if best_acc < curr_acc:
        best_acc = curr_acc
        best_acc_ckpt_name = ckpt_name

    logging.info('Shut down TPU system.')
    sess.run(tpu_shutdown)
    eval_summary_writer.close()

    with gfile.GFile(best_acc_file, 'w') as fout:
      fout.write(f'{best_acc:<6.4f}')
    saver.restore(sess, best_acc_ckpt_name)
    best_ckpt_path = saver.save(
        sess, save_path=os.path.join(best_acc_path, 'ckpt'),
        write_meta_graph=False, write_state=False)
    logging.info(f'Saved best_ckpt `{best_ckpt_path}`')


def main(unused_argv):
  params = flag_utils.build_params_from_flags()

  def _done(message):
    """Write a file `done` to `params.output_dir` to finish compound jobs."""
    with gfile.GFile(os.path.join(params.output_dir, 'done'), 'w') as fout:
      fout.write(message)
      fout.flush()

  def _should_eval():
    if params.use_vizier:
      return True
    name = params.dataset_name.lower()
    small_datasets = ['cifar', 'imagenet_10_percent']
    eval_conditions = [name.startswith(n) for n in small_datasets]
    return any(eval_conditions)
  should_eval = _should_eval()

  for trial_id in range(MAX_RETRIES):
    try:
      train_tpu(params, should_eval=should_eval)
      if not should_eval:
        eval_tpu(params)
      _done('Success')
      break
    except:  # pylint: disable=bare-except
      traceback.print_exc()
      logging.info(f'Failed {trial_id+1} times. Retry!')
      continue
  else:
    _done(f'Job failed after {MAX_RETRIES} retries')


if __name__ == '__main__':
  tf.disable_v2_behavior()
  np.set_printoptions(
      precision=3, suppress=True, threshold=int(1e9), linewidth=160)
  app.run(main)
