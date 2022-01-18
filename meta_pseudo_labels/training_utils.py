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

# pylint: disable=logging-format-interpolation
# pylint: disable=unused-import
# pylint: disable=protected-access
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-long-lambda

r"""Docs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import heapq
import os
import sys
import time
import traceback

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from meta_pseudo_labels import common_utils
from meta_pseudo_labels import data_utils

from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from tensorflow.python.tpu import tpu_feed


MODEL_SCOPE = 'model'


def eval_step_fn(params, model):
  """Build `step_fn` for eval."""
  dtypes = [tf.bfloat16 if params.use_bfloat16 else tf.float32,
            tf.float32, tf.float32]
  batch_size = params.eval_batch_size // params.num_replicas
  image_size = (params.eval_image_size if 'eval_image_size' in params
                else params.image_size)
  shapes = [[batch_size, image_size, image_size, 3],
            [batch_size, params.num_classes],
            [batch_size]]

  if params.use_xla_sharding and params.num_cores_per_replica > 1:
    q = tpu_feed._PartitionedInfeedQueue(
        number_of_tuple_elements=3,
        host_id=0,
        input_partition_dims=[[1, 1, params.num_cores_per_replica, 1],
                              [1, 1], [1]],
        device_assignment=params.device_assignment)
    q.set_tuple_types(dtypes)
    q.set_tuple_shapes(shapes)
    images, labels, mask = q.generate_dequeue_op()
    images = xla_sharding.split(images, 2, params.num_cores_per_replica)
  else:
    with tf.device(tf.tpu.core(0)):
      images, labels, mask = tf.raw_ops.InfeedDequeueTuple(dtypes=dtypes,
                                                           shapes=shapes)

  if len(labels.shape) > 1:  # `labels` is one_hot. turn it to `int.32`
    labels = tf.argmax(labels, axis=-1, output_type=tf.int32)
    labels = tf.expand_dims(labels, axis=-1)
  _ = tf.train.get_or_create_global_step()

  with tf.variable_scope(MODEL_SCOPE):
    logits = model(images, training=False)
    logits = tf.cast(logits, tf.float32)

  return logits, labels, mask


class Supervised(object):
  """Supervised information."""

  def __init__(self):
    step_info = collections.OrderedDict()
    self.step_info = step_info

  def outfeed_signature(self):
    """Returns the sigature of `step_info` as returned by `step_fn`."""
    return self.step_info

  def step_fn(self, params, model):
    """A single step for supervised learning."""

    batch_size = params.train_batch_size // params.num_replicas
    dtypes = [tf.bfloat16 if params.use_bfloat16 else tf.float32, tf.float32]
    shapes = [[batch_size, params.image_size, params.image_size, 3],
              [batch_size, params.num_classes]]

    if params.use_xla_sharding and params.num_cores_per_replica > 1:
      q = tpu_feed._PartitionedInfeedQueue(
          number_of_tuple_elements=2,
          host_id=0,
          input_partition_dims=[[1, 1, params.num_cores_per_replica, 1],
                                [1, 1]],
          device_assignment=params.device_assignment)
      q.set_tuple_types(dtypes)
      q.set_tuple_shapes(shapes)
      images, labels = q.generate_dequeue_op()
      images = xla_sharding.split(images, 2, params.num_cores_per_replica)
    else:
      with tf.device(tf.tpu.core(0)):
        images, labels = tf.raw_ops.InfeedDequeueTuple(dtypes=dtypes,
                                                       shapes=shapes)

    if labels.dtype == tf.int32:
      labels = tf.one_hot(labels, depth=params.num_classes, dtype=tf.float32)
    global_step = tf.train.get_or_create_global_step()

    train_batch_size = tf.cast(params.train_batch_size, tf.float32)
    num_replicas = tf.cast(params.num_replicas, tf.float32)

    with tf.variable_scope(MODEL_SCOPE):
      logits = model(images, training=True)

    if 'noisy_student' in params.dataset_name.lower():
      cross_entropy = labels * tf.nn.log_softmax(logits, axis=-1)
      cross_entropy = tf.reduce_sum(-cross_entropy) / train_batch_size
    else:
      cross_entropy = tf.losses.softmax_cross_entropy(
          onehot_labels=labels, logits=logits,
          label_smoothing=params.label_smoothing,
          reduction=tf.losses.Reduction.SUM) / train_batch_size

    l2_reg_rate = tf.cast(params.weight_decay / params.num_replicas, tf.float32)
    weight_dec = common_utils.get_l2_loss()
    total_loss = cross_entropy + weight_dec * l2_reg_rate

    variables = tf.trainable_variables()
    gradients = tf.gradients(total_loss, variables)
    gradients = [tf.tpu.cross_replica_sum(g) for g in gradients]
    gradients, grad_norm = tf.clip_by_global_norm(gradients, params.grad_bound)

    learning_rate, optimizer = common_utils.get_optimizer(params)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.cond(
        tf.math.is_finite(grad_norm),
        lambda: optimizer.apply_gradients(zip(gradients, variables),
                                          global_step=global_step),
        tf.no_op)
    with tf.control_dependencies(update_ops + [train_op]):
      ema_train_op = common_utils.setup_ema(params,
                                            f'{MODEL_SCOPE}/{model.name}')

    with tf.control_dependencies([ema_train_op]):
      logs = collections.OrderedDict()
      logs['global_step'] = tf.cast(global_step, tf.float32)
      logs['loss/total'] = total_loss
      logs['loss/weight_decay'] = weight_dec / num_replicas
      logs['loss/cross_entropy'] = cross_entropy
      logs['loss/lr'] = tf.identity(learning_rate) / num_replicas
      logs['loss/grad_norm'] = grad_norm / num_replicas

      tensors = [tf.expand_dims(t, axis=0) for t in logs.values()]
      self.step_info = {k: [tf.float32, [1]] for k in logs.keys()}
      outfeed_enqueue_op = tf.cond(
          common_utils.should_log(params),
          lambda: tf.raw_ops.OutfeedEnqueueTuple(inputs=tensors), tf.no_op)
    return outfeed_enqueue_op


class UDA(object):
  """UDA (https://arxiv.org/abs/1904.12848)."""

  def __init__(self):
    self.step_info = collections.OrderedDict()

  def outfeed_signature(self):
    """Returns the sigature of `step_info` as returned by `step_fn`."""
    return self.step_info

  @staticmethod
  def build_uda_cross_entropy(params, model, all_images, l_labels):
    """Compute the UDA loss."""
    train_batch_size = params.train_batch_size
    num_replicas = params.num_replicas
    uda_data = params.uda_data
    batch_size = train_batch_size // num_replicas

    labels = {}
    if l_labels.dtype == tf.int32:  # l_labels is sparse. turn into one_hot
      labels['l'] = tf.one_hot(l_labels, params.num_classes, dtype=tf.float32)
    else:
      labels['l'] = l_labels

    global_step = tf.train.get_or_create_global_step()

    masks = {}
    logits = {}
    cross_entropy = {}
    all_logits = model(all_images, training=True)

    logits['l'], logits['u_ori'], logits['u_aug'] = tf.split(
        all_logits, [batch_size, batch_size*uda_data, batch_size*uda_data], 0)

    # sup loss
    cross_entropy['l'] = tf.losses.softmax_cross_entropy(
        onehot_labels=labels['l'],
        logits=logits['l'],
        label_smoothing=params.label_smoothing,
        reduction=tf.losses.Reduction.NONE)
    probs = tf.nn.softmax(logits['l'], axis=-1)
    correct_probs = tf.reduce_sum(labels['l']*probs, axis=-1)
    r = tf.cast(global_step, tf.float32) / float(params.num_train_steps)
    l_threshold = r * (1. - 1./params.num_classes) + 1. / params.num_classes
    masks['l'] = tf.less_equal(correct_probs, l_threshold)
    masks['l'] = tf.cast(masks['l'], tf.float32)
    masks['l'] = tf.stop_gradient(masks['l'])
    cross_entropy['l'] = tf.reduce_sum(cross_entropy['l']) / float(
        train_batch_size)

    # unsup loss
    labels['u_ori'] = tf.nn.softmax(logits['u_ori'] / params.uda_temp, axis=-1)
    labels['u_ori'] = tf.stop_gradient(labels['u_ori'])

    cross_entropy['u'] = (labels['u_ori'] *
                          tf.nn.log_softmax(logits['u_aug'], axis=-1))
    largest_probs = tf.reduce_max(labels['u_ori'], axis=-1, keepdims=True)
    masks['u'] = tf.greater_equal(largest_probs, params.uda_threshold)
    masks['u'] = tf.cast(masks['u'], tf.float32)
    masks['u'] = tf.stop_gradient(masks['u'])
    cross_entropy['u'] = tf.reduce_sum(-cross_entropy['u']*masks['u']) / float(
        train_batch_size*uda_data)
    return logits, labels, masks, cross_entropy

  def step_fn(self, params, model):
    """Separate implementation."""
    train_batch_size = params.train_batch_size
    num_replicas = params.num_replicas
    batch_size = train_batch_size // num_replicas

    dtypes = [
        tf.bfloat16 if params.use_bfloat16 else tf.float32,
        tf.float32,
        tf.bfloat16 if params.use_bfloat16 else tf.float32,
        tf.bfloat16 if params.use_bfloat16 else tf.float32]
    shapes = [
        [batch_size, params.image_size, params.image_size, 3],
        [batch_size, params.num_classes],
        [batch_size*params.uda_data, params.image_size, params.image_size, 3],
        [batch_size*params.uda_data, params.image_size, params.image_size, 3]]

    if params.use_xla_sharding and params.num_cores_per_replica > 1:
      q = tpu_feed._PartitionedInfeedQueue(
          number_of_tuple_elements=4,
          host_id=0,
          input_partition_dims=[[1, 1, params.num_cores_per_replica, 1],
                                [1, 1],
                                [1, 1, params.num_cores_per_replica, 1],
                                [1, 1, params.num_cores_per_replica, 1],],
          device_assignment=params.device_assignment)
      q.set_tuple_types(dtypes)
      q.set_tuple_shapes(shapes)
      l_images, l_labels, u_images_ori, u_images_aug = q.generate_dequeue_op()
      l_images = xla_sharding.split(l_images, 2,
                                    params.num_cores_per_replica)
      u_images_ori = xla_sharding.split(u_images_ori, 2,
                                        params.num_cores_per_replica)
      u_images_aug = xla_sharding.split(u_images_aug, 2,
                                        params.num_cores_per_replica)
    else:
      with tf.device(tf.tpu.core(0)):
        (l_images, l_labels, u_images_ori,
         u_images_aug) = tf.raw_ops.InfeedDequeueTuple(dtypes=dtypes,
                                                       shapes=shapes)

    all_images = tf.concat([l_images, u_images_ori, u_images_aug], axis=0)
    global_step = tf.train.get_or_create_global_step()
    num_replicas = tf.cast(params.num_replicas, tf.float32)

    with tf.variable_scope(MODEL_SCOPE, reuse=tf.AUTO_REUSE):
      _, _, masks, cross_entropy = UDA.build_uda_cross_entropy(
          params, model, all_images, l_labels)

    l2_reg_rate = tf.cast(params.weight_decay / params.num_replicas, tf.float32)
    weight_dec = common_utils.get_l2_loss()
    uda_weight = params.uda_weight * tf.minimum(
        1., tf.cast(global_step, tf.float32) / float(params.uda_steps))
    total_loss = (cross_entropy['u'] * uda_weight +
                  cross_entropy['l'] +
                  weight_dec * l2_reg_rate)
    variables = tf.trainable_variables()
    gradients = tf.gradients(total_loss, variables)
    gradients = [tf.tpu.cross_replica_sum(g) for g in gradients]
    gradients, grad_norm = tf.clip_by_global_norm(gradients, params.grad_bound)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    learning_rate, optimizer = common_utils.get_optimizer(params)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(zip(gradients, variables),
                                           global_step=global_step)

    with tf.control_dependencies([train_op]):
      ema_train_op = common_utils.setup_ema(
          params, f'{MODEL_SCOPE}/{model.name}')

    with tf.control_dependencies([ema_train_op]):
      logs = collections.OrderedDict()
      logs['global_step'] = tf.cast(global_step, tf.float32)
      logs['loss/total'] = total_loss
      logs['loss/cross_entropy'] = cross_entropy['l']
      logs['loss/lr'] = tf.identity(learning_rate) / num_replicas
      logs['loss/grad_norm'] = tf.identity(grad_norm) / num_replicas
      logs['loss/weight_dec'] = weight_dec / num_replicas

      logs['uda/cross_entropy'] = cross_entropy['u']
      logs['uda/u_ratio'] = tf.reduce_mean(masks['u']) / num_replicas
      logs['uda/l_ratio'] = tf.reduce_mean(masks['l']) / num_replicas
      logs['uda/weight'] = uda_weight / num_replicas

      tensors = [tf.expand_dims(t, axis=0) for t in logs.values()]
      self.step_info = {k: [tf.float32, [1]] for k in logs.keys()}
      outfeed_enqueue_op = tf.cond(
          common_utils.should_log(params),
          lambda: tf.raw_ops.OutfeedEnqueueTuple(inputs=tensors), tf.no_op)
    return outfeed_enqueue_op


class MPL(object):
  """Meta Pseudo Labels."""

  def __init__(self):
    self.step_info = collections.OrderedDict()

  def outfeed_signature(self):
    """Returns the sigature of `step_info` as returned by `step_fn`."""
    return self.step_info

  def step_fn(self, params, model):
    """Separate implementation."""
    train_batch_size = params.train_batch_size
    num_replicas = params.num_replicas
    uda_data = params.uda_data
    batch_size = train_batch_size // num_replicas

    dtypes = [
        tf.bfloat16 if params.use_bfloat16 else tf.float32,
        tf.float32,
        tf.bfloat16 if params.use_bfloat16 else tf.float32,
        tf.bfloat16 if params.use_bfloat16 else tf.float32]
    shapes = [
        [batch_size, params.image_size, params.image_size, 3],
        [batch_size, params.num_classes],
        [batch_size*params.uda_data, params.image_size, params.image_size, 3],
        [batch_size*params.uda_data, params.image_size, params.image_size, 3]]

    if params.use_xla_sharding and params.num_cores_per_replica > 1:
      q = tpu_feed._PartitionedInfeedQueue(
          number_of_tuple_elements=4,
          host_id=0,
          input_partition_dims=[[1, 1, params.num_cores_per_replica, 1],
                                [1, 1],
                                [1, 1, params.num_cores_per_replica, 1],
                                [1, 1, params.num_cores_per_replica, 1],],
          device_assignment=params.device_assignment)
      q.set_tuple_types(dtypes)
      q.set_tuple_shapes(shapes)
      l_images, l_labels, u_images_ori, u_images_aug = q.generate_dequeue_op()
      l_images = xla_sharding.split(l_images, 2,
                                    params.num_cores_per_replica)
      u_images_ori = xla_sharding.split(u_images_ori, 2,
                                        params.num_cores_per_replica)
      u_images_aug = xla_sharding.split(u_images_aug, 2,
                                        params.num_cores_per_replica)
    else:
      with tf.device(tf.tpu.core(0)):
        (l_images, l_labels, u_images_ori,
         u_images_aug) = tf.raw_ops.InfeedDequeueTuple(dtypes=dtypes,
                                                       shapes=shapes)
    global_step = tf.train.get_or_create_global_step()
    num_replicas = tf.cast(params.num_replicas, tf.float32)

    all_images = tf.concat([l_images, u_images_ori, u_images_aug], axis=0)

    # all calls to teacher
    with tf.variable_scope('teacher', reuse=tf.AUTO_REUSE):
      logits, labels, masks, cross_entropy = UDA.build_uda_cross_entropy(
          params, model, all_images, l_labels)

    # 1st call to student
    with tf.variable_scope(MODEL_SCOPE):
      u_aug_and_l_images = tf.concat([u_images_aug, l_images], axis=0)
      logits['s_on_u_aug_and_l'] = model(u_aug_and_l_images, training=True)
      logits['s_on_u'], logits['s_on_l_old'] = tf.split(
          logits['s_on_u_aug_and_l'],
          [u_images_aug.shape[0].value, l_images.shape[0].value], axis=0)

    # for backprop
    cross_entropy['s_on_u'] = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.stop_gradient(tf.nn.softmax(logits['u_aug'], -1)),
        logits=logits['s_on_u'],
        label_smoothing=params.label_smoothing,
        reduction=tf.losses.Reduction.NONE)
    cross_entropy['s_on_u'] = tf.reduce_sum(cross_entropy['s_on_u']) / float(
        train_batch_size*uda_data)

    # for Taylor
    cross_entropy['s_on_l_old'] = tf.losses.softmax_cross_entropy(
        onehot_labels=labels['l'],
        logits=logits['s_on_l_old'],
        reduction=tf.losses.Reduction.SUM)
    cross_entropy['s_on_l_old'] = tf.tpu.cross_replica_sum(
        cross_entropy['s_on_l_old']) / float(train_batch_size)
    shadow = tf.get_variable(
        name='cross_entropy_old', shape=[], trainable=False, dtype=tf.float32)
    shadow_update = tf.assign(shadow, cross_entropy['s_on_l_old'])

    w_s = {}
    g_s = {}
    g_n = {}
    lr = {}
    optim = {}
    w_s['s'] = [w for w in tf.trainable_variables()
                if w.name.lower().startswith(MODEL_SCOPE)]
    g_s['s_on_u'] = tf.gradients(cross_entropy['s_on_u'], w_s['s'])
    # g_s['s_on_u'] = [tf.tpu.cross_replica_sum(g) for g in g_s['s_on_u']]

    lr['s'] = common_utils.get_learning_rate(
        params,
        initial_lr=params.mpl_student_lr,
        num_warmup_steps=params.mpl_student_lr_warmup_steps,
        num_wait_steps=params.mpl_student_lr_wait_steps)
    lr['s'], optim['s'] = common_utils.get_optimizer(
        params, learning_rate=lr['s'])
    optim['s']._create_slots(w_s['s'])  # pylint: disable=protected-access
    update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                  if op.name.startswith(f'train/{MODEL_SCOPE}/')]

    with tf.control_dependencies(update_ops + [shadow_update]):
      g_s['s_on_u'] = common_utils.add_weight_decay(
          params, w_s['s'], g_s['s_on_u'])
      g_s['s_on_u'], g_n['s_on_u'] = tf.clip_by_global_norm(
          g_s['s_on_u'], params.grad_bound)
      train_op = optim['s'].apply_gradients(zip(g_s['s_on_u'], w_s['s']))

      with tf.control_dependencies([train_op]):
        ema_train_op = common_utils.setup_ema(
            params, name_scope=f'{MODEL_SCOPE}/{model.name}')

    # 2nd call to student
    with tf.control_dependencies([ema_train_op]):
      with tf.variable_scope(MODEL_SCOPE, reuse=tf.AUTO_REUSE):
        logits['s_on_l_new'] = model(l_images, training=True)

    cross_entropy['s_on_l_new'] = tf.losses.softmax_cross_entropy(
        onehot_labels=labels['l'],
        logits=logits['s_on_l_new'],
        reduction=tf.losses.Reduction.SUM)
    cross_entropy['s_on_l_new'] = tf.tpu.cross_replica_sum(
        cross_entropy['s_on_l_new']) / float(train_batch_size)

    dot_product = cross_entropy['s_on_l_new'] - shadow
    # dot_product = tf.clip_by_value(
    #     dot_product,
    #     clip_value_min=-params.mpl_dot_product_bound,
    #     clip_value_max=params.mpl_dot_product_bound)
    moving_dot_product = tf.get_variable(
        'moving_dot_product', shape=[], trainable=False, dtype=tf.float32)
    moving_dot_product_update = tf.assign_sub(
        moving_dot_product, 0.01 * (moving_dot_product - dot_product))
    with tf.control_dependencies([moving_dot_product_update]):
      dot_product = dot_product - moving_dot_product
      dot_product = tf.stop_gradient(dot_product)
    cross_entropy['mpl'] = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.stop_gradient(tf.nn.softmax(logits['u_aug'], axis=-1)),
        logits=logits['u_aug'],
        reduction=tf.losses.Reduction.NONE)
    cross_entropy['mpl'] = tf.reduce_sum(cross_entropy['mpl']) / float(
        train_batch_size*uda_data)

    # teacher train op
    uda_weight = params.uda_weight * tf.minimum(
        1., tf.cast(global_step, tf.float32) / float(params.uda_steps))
    teacher_loss = (cross_entropy['u'] * uda_weight +
                    cross_entropy['l'] +
                    cross_entropy['mpl'] * dot_product)
    w_s['t'] = [w for w in tf.trainable_variables() if 'teacher' in w.name]
    g_s['t'] = tf.gradients(teacher_loss, w_s['t'])
    g_s['t'] = common_utils.add_weight_decay(params, w_s['t'], g_s['t'])
    g_s['t'], g_n['t'] = tf.clip_by_global_norm(g_s['t'], params.grad_bound)
    lr['t'] = common_utils.get_learning_rate(
        params,
        initial_lr=params.mpl_teacher_lr,
        num_warmup_steps=params.mpl_teacher_lr_warmup_steps)
    lr['t'], optim['t'] = common_utils.get_optimizer(params,
                                                     learning_rate=lr['t'])

    teacher_train_op = optim['t'].apply_gradients(zip(g_s['t'], w_s['t']),
                                                  global_step=global_step)

    with tf.control_dependencies([teacher_train_op]):
      logs = collections.OrderedDict()
      logs['global_step'] = tf.cast(global_step, tf.float32)

      logs['cross_entropy/student_on_u'] = cross_entropy['s_on_u']
      logs['cross_entropy/student_on_l'] = (cross_entropy['s_on_l_new'] /
                                            num_replicas)
      logs['cross_entropy/teacher_on_u'] = cross_entropy['u']
      logs['cross_entropy/teacher_on_l'] = cross_entropy['l']
      logs['lr/student'] = tf.identity(lr['s']) / num_replicas
      logs['lr/teacher'] = tf.identity(lr['t']) / num_replicas
      logs['mpl/dot_product'] = dot_product / num_replicas
      logs['mpl/moving_dot_product'] = moving_dot_product / num_replicas
      logs['uda/u_ratio'] = tf.reduce_mean(masks['u']) / num_replicas
      logs['uda/l_ratio'] = tf.reduce_mean(masks['l']) / num_replicas
      logs['uda/weight'] = uda_weight / num_replicas

      tensors = [tf.expand_dims(t, axis=0) for t in logs.values()]
      self.step_info = {k: [tf.float32, [1]] for k in logs.keys()}
      def outfeed(tensors):
        with tf.device(tf.tpu.core(params.num_cores_per_replica-1)):
          return tf.raw_ops.OutfeedEnqueueTuple(inputs=tensors)

      outfeed_enqueue_op = tf.cond(
          common_utils.should_log(params), lambda: outfeed(tensors), tf.no_op)

      return outfeed_enqueue_op
