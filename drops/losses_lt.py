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

r"""Loss of long-tail experiments.
"""
import functools
import numpy as np
import tensorflow as tf


def MakeLossFunc(loss_name, samples_per_cls,
                 gamma=0.9, beta=1.0, s=1, tau=1.0):
  """Make a loss function that takes y_true and y_pred (logits) as input."""
  loss_function = loss_name
  if loss_function == 'ce':
    return functools.partial(
        CELoss,
        sample_per_cls=samples_per_cls,
        from_logits=True)

  if loss_function == 'up_ce':
    return functools.partial(
        CELoss,
        sample_per_cls=samples_per_cls,
        from_logits=True)

  if loss_function == 'ldam':
    return functools.partial(
        LDAMLoss,
        sample_per_cls=samples_per_cls,
        gamma=gamma,
        s=s)

  if loss_function == 'focal':
    return functools.partial(
        FocalLoss,
        gamma=gamma,
        sample_per_cls=samples_per_cls,
        from_logits=True)

  if loss_function == 'cb':
    return functools.partial(
        CBLoss,
        sample_per_cls=samples_per_cls,
        beta=beta)

  if loss_function == 'cb_focal':
    return functools.partial(
        CBFocal,
        gamma=gamma,
        beta=beta,
        sample_per_cls=samples_per_cls,
        from_logits=True)

  if loss_function == 'bsm':
    return functools.partial(
        BalancedSoftmax,
        from_logits=True,
        sample_per_cls=samples_per_cls)

  if loss_function == 'logit_adj':
    return functools.partial(
        LogitAdjust,
        sample_per_cls=samples_per_cls,
        tau=tau)

  if loss_function == 'posthoc_ce':
    return functools.partial(
        CELoss,
        sample_per_cls=samples_per_cls,
        from_logits=True)

  if loss_function == 'posthoc':
    return functools.partial(
        LogitAdjust,
        sample_per_cls=samples_per_cls,
        tau=tau)

  if loss_function == 'drops':
    return functools.partial(
        CELoss,
        sample_per_cls=samples_per_cls,
        from_logits=True)
  raise ValueError('Unsupported loss function.')


def CELoss(y_true,
           y_pred,
           sample_per_cls,
           from_logits=False):
  """ce loss.

  Args:
    y_true: True labels, categorical of shape (batch_size,).
    y_pred: logits, a float32 tensor of shape [batch_size,num_classes].
    sample_per_cls: number of samples per class [num_classes].
    from_logits: True if y_pred is not soft-maxed.

  Returns:
    loss: A tensor of shape [batch_size, num_classes].
  """
  labels_oh = MaybeOneHot(y_true, depth=len(sample_per_cls))
  loss = tf.keras.backend.categorical_crossentropy(
      labels_oh, y_pred, from_logits=from_logits)
  return tf.reduce_mean(loss)


def CBLoss(y_true,
           y_pred,
           sample_per_cls,
           beta):
  """Computer class balanced loss for MULTICLASS classification.

  Paper link: https://arxiv.org/pdf/1901.05555.pdf

  Args:
    y_true: True labels, categorical of shape (batch_size,).
    y_pred: logits, a float32 tensor of shape [batch_size,num_classes].
    sample_per_cls: number of samples per class [num_classes].
    beta: A scalar for CBloss hyper-parameter.

  Returns:
    loss: A tensor of shape [batch_size, num_classes].
  """
  batch_size = y_pred.shape[0]
  class_weight = [(1-beta)/(1-beta**i) for i in sample_per_cls]
  class_weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
   # Equations:
   # loss = -[(1-beta) / (1-beta^n_y)] * log(prob_y)
  xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y_true, logits=y_pred)
  loss = xent_loss
  class_weight = tf.gather(class_weight, y_true, axis=0,
                           batch_dims=y_true.shape.rank)
  scaler_weight = float(batch_size) * class_weight / tf.reduce_sum(class_weight)
  loss *= scaler_weight
  return tf.reduce_sum(loss)/batch_size


def CBFocal(y_true,
            y_pred,
            sample_per_cls,
            beta,
            gamma,
            from_logits=False):
  """Computer class balanced loss for MULTICLASS classification.

  Args:
    y_true: True labels, categorical of shape (batch_size,).
    y_pred: logits, a float32 tensor of shape [batch_size,num_classes].
    sample_per_cls: number of samples per class [num_classes].
    beta: A scalar for CBloss hyper-parameter.
    gamma: A scalar for CBloss hyper-parameter.
    from_logits: True if y_pred is not soft-maxed.

  Returns:
    loss: A tensor of shape [batch_size, num_classes].
  """
  batch_size = y_pred.shape[0]
  class_weight = [(1-beta)/(1-beta**i) for i in sample_per_cls]
  class_weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
  if from_logits:
    y_pred_prob = tf.nn.softmax(y_pred, axis=-1)
  else:
    y_pred_prob = y_pred
   # Equations:
   # loss = -[(1-beta) / (1-beta^n_y)] * (1-prob_y)^gamma * log(prob_y)
  xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y_true, logits=y_pred)
  probs = tf.gather(y_pred_prob, y_true, axis=-1, batch_dims=y_true.shape.rank)
  focal_modulation = (1 - probs) ** gamma
  loss = focal_modulation * xent_loss
  class_weight = tf.gather(class_weight, y_true, axis=0,
                           batch_dims=y_true.shape.rank)
  scaler_weight = float(batch_size) * class_weight / tf.reduce_sum(class_weight)
  loss *= scaler_weight
  return tf.reduce_sum(loss)/batch_size


def LDAMLoss(y_true,
             y_pred,
             sample_per_cls,
             gamma,
             s=30):
  """Computer LDAM loss for MULTICLASS classification.

  Args:
    y_true: True labels, categorical of shape (batch_size,).
    y_pred: logits, a float32 tensor of shape [batch_size,num_classes].
    sample_per_cls: number of samples per class [num_classes].
    gamma: A scalar for the re-weighting of hyper-parameter.
    s: hyper-parameter.

  Returns:
    loss: A tensor of shape [batch_size, num_classes].
  """
  num_classes = y_pred.shape[1]
  class_weight = 1.0 / np.sqrt(np.sqrt(sample_per_cls))
  class_weight = class_weight * (gamma / np.max(class_weight))
  class_weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)
  y_true_oh = ConvertToOneHot(y_true, depth=num_classes)
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
  batch_w = tf.matmul(class_weight[None, :], tf.transpose(y_true_oh))
  batch_w = tf.reshape(batch_w, (-1, 1))
  y_pred_m = y_pred - batch_w
  # if condition is true, return y_pred_m[index], otherwise return y_pred[index]
  index_bool = tf.cast(y_true_oh, tf.bool)
  output = tf.where(index_bool, y_pred_m, y_pred)
  logits = output
  loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=y_true_oh, logits=logits*s)
  return tf.reduce_mean(loss)


def FocalLoss(y_true,
              y_pred,
              sample_per_cls,
              gamma,
              from_logits=False):
  """Computer focal loss for MULTICLASS classification.

  Args:
    y_true: True labels, categorical of shape (batch_size,).
    y_pred: logits, a float32 tensor of shape [batch_size,num_classes].
    sample_per_cls: number of samples per class [num_classes].
    gamma: A scalar for focal loss gamma hyper-parameter.
    from_logits: True if y_pred is not soft-maxed.

  Returns:
    loss: A tensor of shape [batch_size, num_classes].
  """
  batch_size = y_pred.shape[0]
  recip_spc = [1/i for i in sample_per_cls]
  class_weight = tf.cast(recip_spc, dtype=tf.float32)
  # Normalizer to ensure that sum of class weights is equal to batch_size (like
  # in ERM)
  class_weight_norm = float(batch_size) * class_weight
  class_weight_norm /= tf.reduce_sum(class_weight)
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
  if from_logits:
    y_pred_prob = tf.nn.softmax(y_pred, axis=-1)
  else:
    y_pred_prob = y_pred
   # Equations:
   # loss = -alpha_y * (1-prob_y)^gamma * log(prob_y)
  xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y_true, logits=y_pred)
  probs = tf.gather(y_pred_prob, y_true, axis=-1, batch_dims=y_true.shape.rank)
  focal_modulation = (1 - probs) ** gamma
  loss = focal_modulation * xent_loss
  class_weight_norm = tf.gather(class_weight_norm, y_true, axis=0,
                                batch_dims=y_true.shape.rank)

  loss *= class_weight_norm
  return tf.reduce_sum(loss)/batch_size


def LogitAdjust(y_true, y_pred, sample_per_cls, tau):
  """Implementation of logit adjustment loss.

  Args:
    y_true: True labels, categorical of shape (batch_size,).
    y_pred: logits, a float32 tensor of shape [batch_size,num_classes].
    sample_per_cls: number of samples per class [num_classes].
    tau: Temperature scaling parameter for the base probabilities.

  Returns:
    A loss function with signature loss(y_true, y_pred).
  """
  spc = tf.cast(sample_per_cls, dtype=tf.float32)
  # Generate class prior (a list of probabilities: P(Y=i))
  spc_norm = spc / tf.reduce_sum(spc)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  y_pred = y_pred + tau * tf.math.log(
      tf.cast(spc_norm + 1e-12, dtype=tf.float32))
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y_true, logits=y_pred)
  return tf.reduce_mean(loss, axis=0)


def BalancedSoftmax(y_true,
                    y_pred,
                    sample_per_cls,
                    from_logits=False):
  """Implementation of balacned softmax https://arxiv.org/abs/2007.10740.

  Args:
    y_true: True labels, categorical of shape (batch_size,).
    y_pred: logits, a float32 tensor of shape [batch_size,num_classes].
    sample_per_cls: number of samples per class [num_classes].
    from_logits: True if y_pred is not soft-maxed.

  Returns:
    loss: A tensor of shape [batch_size, num_classes].
  """
  num_classes = y_pred.shape[1]
  # batch_size = y_pred.shape[0]
  y_true_oh = ConvertToOneHot(y_true, depth=num_classes)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  spc = tf.cast(sample_per_cls, dtype=tf.float32)
  # Generate class prior (a list of probabilities: P(Y=i))
  spc_norm = spc / tf.reduce_sum(spc)
  if from_logits:
    # reweight exponential of logits with class prior
    y_b_exp = tf.exp(y_pred) * spc_norm
    logits_modified = y_b_exp / tf.reduce_sum(y_b_exp, 1, keepdims=True)
  else:
    raise ValueError(
        'please give me logits inputs'
    )
  cce = tf.keras.losses.CategoricalCrossentropy()
  loss = cce(y_true_oh, logits_modified)
  return loss


def ConvertToOneHot(labels, depth):
  if len(labels.shape) > 1:
    return labels
  else:
    return tf.one_hot(labels, depth=depth)


def MaybeOneHot(labels, depth):
  if len(labels.shape) > 1:
    return labels
  else:
    return tf.one_hot(labels, depth=depth)
