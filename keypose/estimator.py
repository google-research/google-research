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

"""Estimator definition using Keras and TF2.

Adapted from 'Discovery of Latent 3D Keypoints via End-to-end
Geometric Reasoning' keypoint network.

Given a 2D image and viewpoint, predict a set of 3D keypoints that
match the target examples.

Can be instance or class specific, depending on training set.
"""

import tensorflow as tf
from tensorflow import estimator as tf_estimator

from keypose import losses as ls
from keypose import nets


def make_decay_function(step, lr_list):
  """Decay function, stepwise linear from 0.1 to 0.01 of max lr.

  Args:
    step: training step.
    lr_list: [start_decay, end_decay, rate_0, rate_1, ...]

  Returns:
    lr_func: decayed lr function.
    lambda: 10-4 * lr_func()
  """

  start, end = lr_list[0], lr_list[1]
  lr_seq = lr_list[2:]
  seq_num = float(len(lr_seq) - 1)
  seq_len = float(end - start) / seq_num

  def decayed_lr():
    if step < start:
      return lr_seq[0]
    elif step > end:
      return lr_seq[-1]
    else:
      step_trunc = tf.cast(step - start, tf.float32)
      num = tf.cast(step_trunc / seq_len, tf.int64)
      lr_start = tf.gather(lr_seq, num)
      lr_end = tf.gather(lr_seq, num + 1)
      step_start = tf.cast(num, tf.float32) * seq_len
      alpha = (step_trunc - step_start) / seq_len
      return alpha * lr_end + (1.0 - alpha) * lr_start

  lr_func = tf.function(func=decayed_lr)
  return lr_func, lambda: 1e-4 * lr_func()


def est_model_fn(features, labels, mode, params):
  """Model function for the estimator.

  Args:
    features: features from tfrecords.
    labels: labels from tfrecords.
    mode: train or eval.
    params: ConfigParams for the model.

  Returns:
    EstimatorSpec for the model.
  """
  print('In est_model_fn')

  is_training = (mode == tf_estimator.ModeKeys.TRAIN)

  step = tf.compat.v1.train.get_or_create_global_step()
  print('Step is:', step)
  print('Mode is:', mode, is_training)
  #  print('labels is:', labels)

  mparams = params.model_params
  model = nets.keypose_model(mparams, is_training)
  #  print('Features:\n', features)
  preds = model(features, training=is_training)
  print('Symbolic predictions:\n', preds)

  if mode == tf_estimator.ModeKeys.PREDICT:
    print('Model_fn is returning from prediction mode')
    return tf_estimator.EstimatorSpec(mode=mode, predictions=preds)

  # Get both the unconditional losses (the None part)
  # and the input-conditional losses (the features part).
  reg_losses = model.get_losses_for(None) + model.get_losses_for(features)
  # Custom loss here.
  print('Calling loss fn')
  print('Reg losses:', reg_losses)
  if reg_losses:
    reg_loss = tf.math.add_n(reg_losses)
  else:
    reg_loss = 0.0
  print('Reg losses summed:', reg_loss)

  kp_loss, uvdw, xyzw = ls.keypose_loss(labels, preds, step, mparams)
  total_loss = kp_loss + mparams.loss_reg * reg_loss
  print('Total loss:', total_loss)
  print('End calling loss fn')

  # Metrics from tf.keras.metrics, for eval only.
  mae_disp_obj = tf.keras.metrics.Mean(name='MAE_disp')
  mae_disp_obj.update_state(ls.disp_error(labels, uvdw, mparams))
  mae_uv_obj = tf.keras.metrics.Mean(name='MAE_uv')
  mae_uv_obj.update_state(ls.uv_error(labels, uvdw, mparams))
  mae_world_obj = tf.keras.metrics.Mean(name='0_MAE_world')
  mae_world_obj.update_state(ls.world_error(labels, xyzw))
  mae_2cm_obj = tf.keras.metrics.Mean(name='MAE_2cm')
  mae_2cm_obj.update_state(ls.lt_2cm_error(labels, xyzw))
  metric_ops = {
      'MAE_disp': mae_disp_obj,
      'MAE_uv': mae_uv_obj,
      'MAE_2cm': mae_2cm_obj,
      '0_MAE_world': mae_world_obj
  }

  # Training stats - these correspond to the eval metrics.
  # Don't know how to use v2 summaries here.
  tf.compat.v1.summary.image(
      'viz_img_L',
      ls.add_keypoints(features['img_L'], preds['uvd']),
      max_outputs=4)
  tf.compat.v1.summary.image(
      'viz_probs',
      tf.expand_dims(tf.reduce_max(preds['prob'], axis=1), axis=-1),
      max_outputs=4)
  tf.compat.v1.summary.scalar(
      'MAE_disp', tf.reduce_mean(ls.disp_error(labels, uvdw, mparams)))
  tf.compat.v1.summary.scalar(
      'MAE_uv', tf.reduce_mean(ls.uv_error(labels, uvdw, mparams)))
  tf.compat.v1.summary.scalar('0_MAE_world',
                              tf.reduce_mean(ls.world_error(labels, xyzw)))
  tf.compat.v1.summary.scalar('MAE_2cm',
                              tf.reduce_mean(ls.lt_2cm_error(labels, xyzw)))
  # TODO(konolige): Add re-ordered prob loss back in.
  #  tf.compat.v1.summary.scalar('Prob_loss',
  #               tf.reduce_mean(ls.keypose_loss_prob(preds['prob'], labels)))
  if is_training:
    tf.compat.v1.summary.scalar(
        'Adjust_proj', ls.adjust_proj_factor(step, mparams.loss_proj_step))

  train_op = None
  if is_training:
    # Using tf.keras.optimizers.
    # NOTE: unlike in the V2 docs, lr_func is a zero-arg function that
    #   must access the optimizer step itself, rather than having it passed in.
    # decay param in Adam is 1 / (1 + decay * step).  Use custom decay.
    lr_func, _ = make_decay_function(step, params.learning_rate)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_func, epsilon=1e-8, decay=0.0, clipnorm=5.0)

    # Manually assign tf.compat.v1.global_step variable to optimizer.iterations
    # to make tf.compat.v1.train.global_step increase correctly.
    # This assignment is a must for any `tf.train.SessionRunHook` specified in
    # estimator, as SessionRunHooks rely on global step.
    optimizer.iterations = step
    # Get both the unconditional updates (the None part)
    # and the input-conditional updates (the features part).
    update_ops = model.get_updates_for(None) + model.get_updates_for(features)
    # Compute the minimize_op.

    print(tf.compat.v1.global_variables(scope='batch'))
    train_vars = model.trainable_variables
    print('Trainable vars:\n', train_vars)
    print('Trainable flags:\n', [(x.name, x.trainable) for x in train_vars])
    minimize_op = optimizer.get_updates(total_loss, train_vars)[0]
    train_op = tf.group(minimize_op, *update_ops)
    tf.compat.v1.summary.scalar('Learning_rate', optimizer.lr)

  return tf_estimator.EstimatorSpec(
      mode=mode,
      predictions=preds,
      loss=total_loss,
      train_op=train_op,
      eval_metric_ops=metric_ops)
