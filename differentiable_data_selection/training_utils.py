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
# pylint: disable=protected-access
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-long-lambda

r"""Train and Eval drivers."""

import collections

import tensorflow.compat.v1 as tf

from differentiable_data_selection import common_utils


MODEL_SCOPE = 'model'
SCORE_SCOPE = 'score'


def eval_step_fn(params, model):
  """Build `step_fn` for eval."""
  images, labels, mask = tf.raw_ops.InfeedDequeueTuple(
      dtypes=params.eval_dtypes, shapes=params.eval_shapes)

  if len(labels.shape) > 1:  # `labels` is one_hot. turn it to `int.32`
    labels = tf.argmax(labels, axis=-1, output_type=tf.int32)
    labels = tf.expand_dims(labels, axis=-1)
  _ = tf.train.get_or_create_global_step()

  with tf.variable_scope(MODEL_SCOPE):
    logits = model(images, training=False)
    logits = tf.cast(logits, tf.float32)

  return logits, labels, mask


class Supervised(object):
  """Supervised learning."""

  def __init__(self):
    step_info = collections.OrderedDict()
    self.step_info = step_info

  def outfeed_signature(self):
    """Returns the sigature of `step_info` as returned by `step_fn`."""
    return self.step_info

  def step_fn(self, params, model):
    """A single step for supervised learning."""
    images, labels = tf.raw_ops.InfeedDequeueTuple(dtypes=params.train_dtypes,
                                                   shapes=params.train_shapes)

    if labels.dtype == tf.int32:
      labels = tf.one_hot(labels, depth=params.num_classes, dtype=tf.float32)
    global_step = tf.train.get_or_create_global_step()

    train_batch_size = tf.cast(params.train_batch_size, tf.float32)
    num_replicas = tf.cast(params.num_replicas, tf.float32)

    with tf.variable_scope(MODEL_SCOPE):
      logits = model(images, training=True)

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


class DDS(object):
  """Trainer with a pre-trained scorer: https://arxiv.org/pdf/1911.10088.pdf."""

  def __init__(self):
    step_info = collections.OrderedDict()
    self.step_info = step_info

  def outfeed_signature(self):
    """Returns the sigature of `step_info` as returned by `step_fn`."""
    return self.step_info

  def step_fn(self, params, model):
    """A single step for supervised learning."""
    (train_images, train_labels, valid_images,
     valid_labels) = tf.raw_ops.InfeedDequeueTuple(
         dtypes=params.train_dtypes, shapes=params.train_shapes)

    if train_labels.dtype == tf.int32:
      train_labels = tf.one_hot(train_labels, depth=params.num_classes,
                                dtype=tf.float32)
    if valid_labels.dtype == tf.int32:
      valid_labels = tf.one_hot(valid_labels, depth=params.num_classes,
                                dtype=tf.float32)
    global_step = tf.train.get_or_create_global_step()

    num_replicas = tf.cast(params.num_replicas, tf.float32)

    with tf.variable_scope(MODEL_SCOPE):
      train_logits = model(train_images, training=True)

    with tf.variable_scope(SCORE_SCOPE):
      score_logits = model(train_images, training=False, return_scores=True)
      score_m = tf.tpu.cross_replica_sum(tf.reduce_sum(score_logits))
      score_m = tf.stop_gradient(score_m) / float(params.num_replicas)
      score_e = tf.exp(score_logits - score_m)
      score_z = tf.tpu.cross_replica_sum(tf.reduce_sum(score_e))
      score_probs = score_e / score_z

    # train the main model
    cross_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels=train_labels,
        logits=train_logits,
        label_smoothing=params.label_smoothing,
        reduction=tf.losses.Reduction.NONE)
    cross_entropy = tf.reduce_sum(cross_entropy * tf.stop_gradient(score_probs))

    l2_reg_rate = tf.cast(params.weight_decay / params.num_replicas, tf.float32)
    weight_dec = common_utils.get_l2_loss(excluded_keywords=[SCORE_SCOPE])
    total_loss = cross_entropy + weight_dec * l2_reg_rate

    model_variables = [
        v for v in tf.trainable_variables() if MODEL_SCOPE in v.name]
    train_gradients = tf.gradients(total_loss, model_variables)
    train_gradients = [tf.tpu.cross_replica_sum(g) for g in train_gradients]
    train_gradients, grad_norm = tf.clip_by_global_norm(
        train_gradients, params.grad_bound)

    learning_rate, optimizer = common_utils.get_optimizer(params)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.cond(
        tf.math.is_finite(grad_norm),
        lambda: optimizer.apply_gradients(
            zip(train_gradients, model_variables), global_step=global_step),
        tf.no_op)
    with tf.control_dependencies(update_ops + [train_op]):
      ema_train_op = common_utils.setup_ema(params,
                                            f'{MODEL_SCOPE}/{model.name}')

    with tf.control_dependencies([ema_train_op]):
      with tf.variable_scope(MODEL_SCOPE, reuse=True):
        valid_logits = model(valid_images, training=False)
        valid_cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=valid_labels,
            logits=valid_logits,
            reduction=tf.losses.Reduction.MEAN) / float(params.num_replicas)
        valid_gradients = tf.gradients(valid_cross_entropy, model_variables)
        valid_gradients = [tf.tpu.cross_replica_sum(g) for g in valid_gradients]

      dot_product = tf.add_n([
          tf.reduce_sum(g_t*g_v)
          for g_t, g_v in zip(train_gradients, valid_gradients)])
      dot_product = tf.stop_gradient(dot_product)
      dot_product_avg = tf.get_variable(
          name='dot_product_avg', shape=[], trainable=False)
      dot_product_update = tf.assign_sub(
          dot_product_avg, 0.01 * (dot_product_avg - dot_product))
      with tf.control_dependencies([dot_product_update]):
        dot_product = tf.identity(dot_product - dot_product_avg)

    # trains the scorer.
    score_entropy = tf.reduce_sum(-score_probs * tf.math.log(score_probs))
    score_entropy = tf.tpu.cross_replica_sum(score_entropy) / float(
        valid_images.shape[0].value)
    score_variables = [
        v for v in tf.trainable_variables() if SCORE_SCOPE in v.name]
    score_gradients = tf.gradients(dot_product*score_entropy, score_variables)
    score_gradients = [tf.tpu.cross_replica_sum(g) for g in score_gradients]
    score_optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params.scorer_lr, use_locking=True)
    score_train_op = tf.cond(
        global_step < params.scorer_wait_steps,
        tf.no_op,
        lambda: score_optimizer.apply_gradients(
            zip(score_gradients, score_variables)))

    with tf.control_dependencies([score_train_op]):
      logs = collections.OrderedDict()
      logs['global_step'] = tf.cast(global_step, tf.float32)

      logs['model/total'] = total_loss
      logs['model/weight_decay'] = weight_dec / num_replicas
      logs['model/cross_entropy'] = cross_entropy
      logs['model/lr'] = tf.identity(learning_rate) / num_replicas
      logs['model/grad_norm'] = grad_norm / num_replicas

      logs['score/dot_product'] = dot_product / num_replicas
      logs['score/dot_product_avg'] = dot_product_avg / num_replicas
      logs['score/entropy'] = score_entropy
      logs['score/p_min'] = tf.reduce_min(score_probs) / num_replicas
      logs['score/p_max'] = tf.reduce_max(score_probs) / num_replicas

      tensors = [tf.expand_dims(t, axis=0) for t in logs.values()]
      self.step_info = {k: [tf.float32, [1]] for k in logs.keys()}
      outfeed_enqueue_op = tf.cond(
          common_utils.should_log(params),
          lambda: tf.raw_ops.OutfeedEnqueueTuple(inputs=tensors), tf.no_op)
    return outfeed_enqueue_op
