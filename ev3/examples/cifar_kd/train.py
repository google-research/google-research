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

"""CIFAR-10/100 example."""

import functools as ft

from absl import logging
from clu import metric_writers
from flax.training import common_utils
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from ev3 import base
from ev3 import decide
from ev3 import optimize
from ev3 import propose
from ev3.examples.cifar_kd import input_pipeline
from ev3.model_history import decide as model_history_decide
from ev3.model_history import eval_util as model_history_eval_util
from ev3.model_history import optimize as model_history_optimize
from ev3.model_history import propose as model_history_propose
from ev3.model_history import struct as model_history_struct
from ev3.model_history import test_util as model_history_test_util
from ev3.utils import data_util
from ev3.utils import eval_util
from ev3.utils import nn_expansion_util
from ev3.utils import nn_util

# import flax
# from vision_transformer.vit_jax import models as vit_models
# from vision_transformer.vit_jax.configs import vit as vit_config


def cross_entropy_loss(logits, labels):
  log_softmax_logits = jax.nn.log_softmax(logits)
  num_classes = log_softmax_logits.shape[-1]
  one_hot_labels = common_utils.onehot(labels, num_classes)
  return -jnp.sum(one_hot_labels * log_softmax_logits) / labels.size


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  acc = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'acc': acc,
  }
  return metrics


def eval_step(model, batch):
  logits, labels = eval_util.basic_pred_label_extractor(
      model.params, batch, model.graph
  )
  return compute_metrics(logits, labels)


def get_apply_fn(vit_model):
  def vit_apply_fn(params, images, train_=False):
    resized_shape = images.shape[:1] + (384, 384) + images.shape[3:]
    resized_images = jax.image.resize(images, resized_shape, method='bilinear')
    return vit_model.apply(params, resized_images, train=train_)

  return vit_apply_fn


def kld(x, y, temp=1, axis=-1):
  return jnp.mean(
      temp
      * jnp.sum(
          jax.nn.softmax(x / temp, axis=axis)
          * (
              jax.nn.log_softmax(x / temp, axis=axis)
              - jax.nn.log_softmax(y / temp, axis=axis)
          ),
          axis=axis,
      )
  )


def get_loss_fn():
  """Loss function for the student model."""

  def loss_fn(
      student_params,
      student_model_graph,
      loss_state,
      batch,
      teacher_params,
      teacher_model_graph,
  ):
    s_logits, batch_stats, _ = eval_util.basic_pred_label_extractor_bn(
        student_params, batch, student_model_graph
    )
    t_logits = teacher_model_graph.apply_fn(
        teacher_params, batch['feature'], train=False
    )
    t_logits = jax.lax.stop_gradient(t_logits)
    loss = kld(s_logits, t_logits, loss_state.temperature)
    return loss, batch_stats

  return jax.jit(loss_fn)


def get_loss_fn_vit():
  """Loss function for the student model."""

  def loss_fn(
      student_params,
      student_model_graph,
      loss_state,
      batch,
  ):
    s_logits, batch_stats, _ = eval_util.basic_pred_label_extractor_bn(
        student_params, batch, student_model_graph
    )
    t_logits = batch['label']
    loss = kld(s_logits, t_logits, loss_state.temperature)
    return loss, batch_stats

  return jax.jit(loss_fn)


def get_metric_fns():
  """Accuracy metric functions."""

  def accuracy_fn_1(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return preds == labels

  vec_accuracy_fn = eval_util.vectorize_eval_fn(accuracy_fn_1)
  mean_accuracy_fn = eval_util.get_mean_eval_fn(accuracy_fn_1)

  @jax.jit
  def vec_metric_fn(
      model_params,
      model_graph,
      batch,
  ):
    logits, labels = eval_util.basic_pred_label_extractor(
        model_params, batch, model_graph
    )
    return vec_accuracy_fn(logits, labels)

  @jax.jit
  def mean_metric_fn(
      model_params,
      model_graph,
      batch,
  ):
    logits, labels = eval_util.basic_pred_label_extractor(
        model_params, batch, model_graph
    )
    return mean_accuracy_fn(logits, labels)

  return vec_metric_fn, mean_metric_fn


def get_optimizers(
    config,
):
  """Get optimizers from config."""
  tx_list = []
  for tx, lr in zip(config.tx_list, config.lr_list):
    schedule_fn = None
    if 'cosine_steps' in lr:
      schedule_fn = optax.cosine_decay_schedule(
          init_value=lr['learning_rate'],
          decay_steps=lr['cosine_steps'],
      )
    if 'warmup_steps' in lr:
      if schedule_fn is None:
        schedule_fn = optax.constant_schedule(lr['learning_rate'])
      warmup_fn = optax.linear_schedule(
          init_value=0.0,
          end_value=lr['learning_rate'],
          transition_steps=lr['warmup_steps'],
      )
      schedule_fn = optax.join_schedules(
          schedules=[warmup_fn, schedule_fn],
          boundaries=[lr['warmup_steps']],
      )
    if schedule_fn is None:
      schedule_fn = lr['learning_rate']
    if tx['name'] == 'sgd':
      tx_list.append(
          optax.inject_hyperparams(optax.sgd)(
              learning_rate=schedule_fn, **tx['kwargs']
          )
      )
    elif tx['name'] == 'adamw':
      tx_list.append(
          optax.inject_hyperparams(optax.adamw)(
              learning_rate=schedule_fn, **tx['kwargs']
          )
      )
    else:
      raise NotImplementedError(tx['name'])
  return tx_list


def get_teacher_data_iterator(fname):
  with open(fname, 'rb') as in_f:
    data = np.load(in_f)
    all_data_with_uint8 = {
        'feature': data['feature'],
        'label': data['label'],
    }
  all_data = all_data_with_uint8.copy()
  all_data['feature'] = all_data['feature'].astype(np.float32)
  all_data['feature'] = all_data['feature'] * 2 / 255 - 1

  return data_util.NumpyDataIterator(
      all_data=all_data,
      batch_size=128,
      n_all_data=len(all_data['label']),
  )


def train(config, workdir):
  """Train model."""
  # if jax.host_count() > 1:
  #   raise ValueError(
  #       'CIFAR-10 example should not be run on more than 1 host (for now)'
  #   )

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )

  num_steps = config.num_steps
  batch_size = config.batch_size
  traj_length = config.traj_length
  traj_mul_factor = config.traj_mul_factor
  steps_before_expansion = config.steps_before_expansion
  num_optimize_batches = config.num_optimize_batches
  num_decide_batches = config.num_decide_batches
  rand_key = random.PRNGKey(config.seed)

  # TODO(li): add parallel training for EV3.
  # if batch_size % jax.device_count() > 0:
  #   raise ValueError('Batch size must be divisible by the number of devices')
  # device_batch_size = batch_size // jax.device_count()

  # Load dataset.
  cifar_input_helper = input_pipeline.GenericInputPipeline(config.dataset_name)

  # propose_data_iter = data_util.TFDataIterator(
  #     ds_name=config.dataset_name,
  #     batch_size=batch_size,
  #     process_fn=cifar_input_helper.process_propose_samples,
  # )
  data_file = '/path/to/data/vit_cifar100/augmented_uint8_images_with_vit_logits_10000_batches.npz'
  propose_data_iter = get_teacher_data_iterator(data_file)
  optimize_data_iter = data_util.TFDataIterator(
      ds_name=config.dataset_name,
      batch_size=batch_size,
      process_fn=cifar_input_helper.process_optimize_samples,
  )
  decide_data_iter = data_util.TFDataIterator(
      ds_name=config.dataset_name,
      batch_size=batch_size,
      process_fn=cifar_input_helper.process_decide_samples,
  )
  eval_data_iter = data_util.TFDataIterator(
      ds_name=config.dataset_name,
      batch_size=100,
      process_fn=cifar_input_helper.process_test_sample,
      split='test',
      shuffle=False,
  )

  steps_per_eval = cifar_input_helper.EVAL_IMAGES // 100

  # Create the teacher model.
  # vit_model = vit_models.VisionTransformer(
  #     num_classes=cifar_input_helper.NUM_CLASSES,
  #     **vit_config.get_config('b16,' + config.dataset_name)['model'],
  # )
  # if config.dataset_name == 'cifar100':
  #   checkpoint = flax.training.checkpoints.restore_checkpoint(
  #       path_to_checkpoint, None
  #   )
  # else:
  #   raise NotImplementedError(config.dataset_name)
  # teacher_params = dict(params=checkpoint[0])
  # teacher_model_graph = model_history_struct.ModelGraph(
  #     nn_model=vit_model,
  #     apply_fn=get_apply_fn(vit_model),
  #     expand_fn=None,
  # )

  # Create an archive to store trained models.
  archive = {}

  pass_idx = 0
  step = 0
  while pass_idx < config.num_passes:
    # Create loss functions.
    loss_states = [
        model_history_test_util.XEntLossState(temperature=t)
        for t in config.loss_temp_list
    ]
    if pass_idx == 0:
      # Only distill from teacher.
      loss_fn_list = [
          get_loss_fn_vit(),
      ]
    else:
      # Distill from teacher and students.
      loss_fn_list = [
          get_loss_fn_vit(),
      ]
      for teacher in archive.values():
        loss_fn_list.append(
            ft.partial(
                get_loss_fn(),
                teacher_params=teacher['params'],
                teacher_model_graph=teacher['model_graph'],
            )
        )
    n_losses = len(loss_fn_list)
    n_loss_states = len(loss_states)
    loss_fn_list = tuple(loss_fn_list * n_loss_states)
    loss_states = tuple(
        sum(([loss_state] * n_losses for loss_state in loss_states), [])
    )

    # Metric.
    vec_metric_fn, _ = get_metric_fns()

    # Create the student model.
    if pass_idx == 0 or config.init_model_every_pass:
      # Initialize the model.
      nn_model = nn_util.get_resnet_model(
          config.nn_model,
          num_classes=cifar_input_helper.NUM_CLASSES,
          num_filters=config.nn_model_num_filters,
          stage_sizes=[config.nn_model_stage_size] * 3,
      )
      if config.nn_expand_fn:
        nn_expand_fn = nn_expansion_util.get_expand_fn(config.nn_expand_fn)
      else:
        nn_expand_fn = None

      batch = next(propose_data_iter)
      x = batch['feature']
      (param_key, model_key, rand_key) = jax.random.split(rand_key, 3)

      params = nn_model.init(param_key, x)
      model_graph = model_history_struct.ModelGraph(
          nn_model=nn_model,
          apply_fn=nn_model.apply,
          expand_fn=nn_expand_fn,
          expand_kwargs=config.nn_expand_kwargs,
      )
      model = model_history_struct.Model(
          graph=model_graph,
          params=params,
          stable_params=params,
          rand_key=model_key,
          history_max_entries=-1,
      )
    else:
      # Use the smallest model in archive.
      (model_key, rand_key) = jax.random.split(rand_key, 2)
      smallest_idx = min(archive.keys())
      params = archive[smallest_idx]['params']
      model_graph = archive[smallest_idx]['model_graph']
      model = model_history_struct.Model(
          graph=model_graph,
          params=params,
          stable_params=params,
          rand_key=model_key,
          history_max_entries=-1,
      )

    tx_list = get_optimizers(config)
    p_state = model_history_struct.ProposeState(
        data_iter=propose_data_iter,
        loss_fn_list=loss_fn_list,
        loss_states=tuple(loss_states),
        trajectory_length=traj_length,
        tx_list=tuple(tx_list),
        traj_mul_factor=traj_mul_factor,
        has_aux=True,
    )
    o_state = model_history_struct.OptimizeState(
        data_iter=optimize_data_iter,
        metric_fn_list=tuple([vec_metric_fn]),
        ucb_alpha=config.optimize_ucb_alpha,
    )
    d_state = model_history_struct.DecideState(
        data_iter=decide_data_iter,
        metric_fn_list=tuple([vec_metric_fn]),
        ucb_alpha=config.decide_ucb_alpha,
    )

    p_tx = propose.get_propose_tx(
        p_state,
        propose_init_fn=model_history_propose.propose_init,
        propose_update_fn=ft.partial(
            model_history_propose.propose_update,
            generate_an_update_fn=model_history_propose.generate_an_update_bn,
        ),
    )
    o_tx = optimize.get_optimize_tx(
        o_state,
        optimize_init_fn=model_history_optimize.optimize_init,
        optimize_update_fn=ft.partial(
            model_history_optimize.optimize_update,
            evaluate_updates_fn=ft.partial(
                model_history_eval_util.evaluate_updates_batches,
                n_batches=num_optimize_batches,
            ),
        ),
    )
    if config.skip_decide:
      decide_update_fn = ft.partial(
          model_history_decide.trival_decide_update,
          evaluate_updates_fn=ft.partial(
              model_history_eval_util.evaluate_updates_batches,
              n_batches=num_decide_batches,
          ),
      )
    elif config.skip_decide_with_expansion:
      decide_update_fn = ft.partial(
          model_history_decide.trival_decide_update_with_expansion,
          evaluate_updates_fn=ft.partial(
              model_history_eval_util.evaluate_updates_batches,
              n_batches=num_decide_batches,
          ),
          update_model_graph_fn=ft.partial(
              model_history_decide.update_model_graph,
              history_tracking_length=steps_before_expansion,
          ),
      )
    else:
      decide_update_fn = ft.partial(
          model_history_decide.decide_update,
          evaluate_updates_fn=ft.partial(
              model_history_eval_util.evaluate_updates_batches,
              n_batches=num_decide_batches,
          ),
          update_model_graph_fn=ft.partial(
              model_history_decide.update_model_graph,
              history_tracking_length=steps_before_expansion,
          ),
          select_best_updates_fn=ft.partial(
              model_history_decide.select_best_update,
          ),
      )
    d_tx = decide.get_decide_tx(
        d_state,
        decide_init_fn=model_history_decide.decide_init,
        decide_update_fn=decide_update_fn,
    )

    tx = optax.chain(p_tx, o_tx, d_tx)
    state = tx.init(model)

    # Gather metrics.
    while True:
      step += traj_length  # only correct when traj_mul_factor=1.

      # Train step.
      model_update, state = tx.update(None, state, model)
      model = model + model_update
      if model.graph.nn_model.stage_sizes[0] > 8:
        break

      model_summary = {
          'resnet_num_filters': model.graph.nn_model.num_filters,
          'resnet_stage_size': model.graph.nn_model.stage_sizes[0],
          'resnet_num_params': sum(
              x.size for x in jax.tree.leaves(model.params)
          ),
          'pass_idx': pass_idx,
      }
      writer.write_scalars(step, model_summary)

      train_summary = {
          'train_'
          + i: model_update.logs[i][model_update.logs['best_param_key']]
          for i in ['lcb', 'ucb', 'acc']
      }
      train_summary = train_summary | {
          'train_selected_loss_idx': model_update.logs['selected_loss_idx']
      }
      writer.write_scalars(step, train_summary)

      # Evaluation.
      eval_summary = {'eval_acc': [], 'eval_loss': []}
      for _ in range(steps_per_eval):
        eval_batch = next(eval_data_iter)
        eval_metrics = eval_step(model, eval_batch)
        eval_summary['eval_acc'].append(eval_metrics['acc'])
        eval_summary['eval_loss'].append(eval_metrics['loss'])
      eval_summary = {
          'eval_acc': jnp.stack(eval_summary['eval_acc']).mean(),
          'eval_loss': jnp.stack(eval_summary['eval_loss']).mean(),
      }
      writer.write_scalars(step, eval_summary)
      writer.flush()

      # Log epoch summary.
      logging.info(
          'STEP %d: PASS=%d, resnet_stage_size=%d, TRAIN acc= %.2f, lcb=%.2f,'
          ' ucb=%.2f, EVAL acc= %.2f, loss=%.6f',
          step,
          pass_idx,
          model.graph.nn_model.stage_sizes[0],
          train_summary['train_acc'] * 100.0,
          train_summary['train_lcb'] * 100.0,
          train_summary['train_ucb'] * 100.0,
          eval_summary['eval_acc'] * 100.0,
          eval_summary['eval_loss'],
      )

      # Save model to archive.
      archive_idx = model_summary['resnet_stage_size']
      if archive_idx not in archive:
        archive[archive_idx] = {
            'params': model.params,
            'model_graph': model.graph,
            'pass_idx': pass_idx,
            'train_acc': train_summary['train_acc'],
            'eval_acc': eval_summary['eval_acc'],
        }
        logging.info(
            'Updating archive: # Pass=%d, Stage size=%d, EVAL acc= %.2f',
            pass_idx,
            archive_idx,
            eval_summary['eval_acc'] * 100.0,
        )
      elif archive[archive_idx]['train_acc'] < train_summary['train_acc']:
        archive[archive_idx] = {
            'params': model.params,
            'model_graph': model.graph,
            'pass_idx': pass_idx,
            'train_acc': train_summary['train_acc'],
            'eval_acc': eval_summary['eval_acc'],
        }
        logging.info(
            'Updating archive: # Pass=%d, Stage size=%d, EVAL acc= %.2f',
            pass_idx,
            archive_idx,
            eval_summary['eval_acc'] * 100.0,
        )

      if step >= num_steps and num_steps >= 0:
        break

    logging.info('Finished pass %d', pass_idx)
    for archive_idx in archive:
      logging.info(
          'Pass=%d, Stage size=%d, EVAL acc= %.2f',
          pass_idx,
          archive_idx,
          archive[archive_idx]['eval_acc'] * 100.0,
      )
    pass_idx += 1
