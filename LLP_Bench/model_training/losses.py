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

"""Loss functions."""

from typing import Callable

from network import CustomModelRegression
import tensorflow as tf
import tensorflow_probability as tfp


tfk = tf.keras
tfkl = tf.keras.layers


def dllp_loss(
    batch_size,
    instance_loss,
    y,
    y_pred,
):
  """DLLP loss."""
  true_list = []
  pred_list = []
  start = 0
  for _ in range(batch_size):
    bag_size = int(y[start, 0])
    avg_true_label = y[start + 1, 0] / y[start, 0]
    true_list.append(avg_true_label)
    avg_pred_label = tf.reduce_mean(y_pred[start : start + bag_size, :])
    pred_list.append(avg_pred_label)
    start = start + bag_size
  return instance_loss(
      tf.convert_to_tensor(true_list), tf.convert_to_tensor(pred_list)
  )


def dllp_loss_graph(
    batch_size,
    instance_loss,
    x,
    y,
    y_pred,
):
  """DLLP loss for graph execution."""
  del x
  y_sg = tf.stop_gradient(y)
  loss = tf.zeros((), dtype=tf.float32)
  start = tf.zeros((), dtype=tf.int32)
  for _ in range(batch_size):
    bag_size = tf.cast(tf.gather(y_sg, start, axis=0)[0], dtype=tf.int32)
    avg_true_label = (
        tf.gather(y_sg, start + 1, axis=0)[0]
        / tf.gather(y_sg, start, axis=0)[0]
    )
    avg_pred_label = tf.reduce_mean(
        tf.gather(y_pred, tf.range(start, start + bag_size), axis=0)
    )
    loss += instance_loss(
        tf.convert_to_tensor([avg_true_label]),
        tf.convert_to_tensor([avg_pred_label]),
    )
    start += bag_size
  return loss / batch_size


def sink(
    a,
    b,
    # pylint: disable=invalid-name
    M,
    reg = 0.2,
    max_iter = 1000,
    error_thresh = 1e-9,
):
  """Sinkhorn algorithm in tensorflow."""
  a = tf.expand_dims(a, axis=1)
  b = tf.expand_dims(b, axis=1)
  u = tf.ones(shape=(tf.shape(a)[0], 1), dtype=tf.float32) / tf.cast(
      tf.shape(a)[0], dtype=tf.float32
  )
  v = tf.ones(shape=(tf.shape(b)[0], 1), dtype=tf.float32) / tf.cast(
      tf.shape(b)[0], dtype=tf.float32
  )
  kernel = tf.exp(-M / reg)
  kernel_p = (1.0 / a) * kernel
  counter = tf.constant(0)
  err = tf.constant(1.0)

  def cond(counter, u, v, err):
    del u, v
    return tf.logical_and(
        counter < max_iter,
        err > error_thresh,
    )

  def err_fn():
    return tf.pow(
        tf.norm(tf.reduce_sum(u * (kernel * tf.squeeze(v))) - b, ord=1), 2
    )

  def default_err():
    return err

  def loop_func(counter, u, v, err):
    del v, err
    v = tf.math.divide(b, tf.matmul(tf.transpose(kernel, (1, 0)), u))
    u = 1.0 / tf.matmul(kernel_p, v)
    err = tf.cond(tf.equal(counter % 10, 0), err_fn, default_err)
    counter = tf.add(counter, 1)
    return counter, u, v, err

  _, u, v, _ = tf.while_loop(cond, loop_func, loop_vars=[counter, u, v, err])
  return tf.reshape(u, (-1, 1)) * kernel * tf.reshape(v, (1, -1))


def get_pseudo_labels(
    hard_labels,
    reg,
    bag_size,
    avg_true_label,
    preds,
):
  """Pseudo labels for ER-OT."""
  p = tf.stack([1 - avg_true_label, avg_true_label])
  b = tf.ones(bag_size, dtype=tf.float32) / tf.cast(bag_size, dtype=tf.float32)
  # pylint: disable=invalid-name
  P = tf.stack([1 - preds, preds])
  Q = sink(p, b, -tf.math.log(P), reg)
  pl = tf.cast(bag_size, dtype=tf.float32) * Q[1, :]
  if hard_labels:
    pl = (tf.math.sign(pl - 0.5) + 1.0) / 2.0
  return pl


def erot_loss(
    hard_labels,
    reg,
    batch_size,
    instance_loss,
    x,
    y,
    y_pred,
):
  """EROT loss."""
  del x
  y_sg = tf.stop_gradient(y)
  loss = tf.zeros((), dtype=tf.float32)
  start = tf.zeros((), dtype=tf.int32)
  for _ in range(batch_size):
    bag_size = tf.cast(tf.gather(y_sg, start, axis=0)[0], dtype=tf.int32)
    avg_true_label = (
        tf.gather(y_sg, start + 1, axis=0)[0]
        / tf.gather(y_sg, start, axis=0)[0]
    )
    preds = tf.reshape(
        tf.gather(y_pred, tf.range(start, start + bag_size), axis=0),
        shape=(-1,),
    )
    pseudo_labels = tf.stop_gradient(
        get_pseudo_labels(hard_labels, reg, bag_size, avg_true_label, preds)
    )
    loss += instance_loss(pseudo_labels, preds)
    start += bag_size
  return loss / batch_size


def genbags_loss(
    mean,
    covariance_mat,
    batch_size,
    block_size,
    num_gen_bags_per_block,
    y,
    y_pred,
):
  """Genbags loss."""
  start = 0
  loss = 0.0
  for _ in range(batch_size // block_size):
    wts = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=covariance_mat
    ).sample(num_gen_bags_per_block)
    list_of_diffs = []
    for _ in range(block_size):
      bag_size = int(y[start, 0])
      sum_true_label = float(y[start + 1, 0])
      list_of_diffs.append(
          tf.reduce_sum(y_pred[start : start + bag_size, :]) - sum_true_label
      )
      start = start + bag_size
    loss = loss + tf.square(
        tf.norm(
            tf.linalg.matvec(wts, tf.convert_to_tensor(list_of_diffs)), ord=2
        )
    )
  return tf.convert_to_tensor(loss)


def easy_llp_loss(
    batch_size,
    p,
    instance_loss,
    y,
    y_pred,
):
  """Easy-LLP loss."""
  p = tf.convert_to_tensor(p, dtype=tf.float32)
  start = 0
  loss = 0.0
  for _ in range(batch_size):
    bag_size = int(y[start, 0])
    if bag_size != 0:
      avg_true_label = tf.cast(y[start + 1, 0], tf.float32) / tf.cast(
          y[start, 0], tf.float32
      )
      idx = tf.random.uniform(shape=[], maxval=bag_size, dtype=tf.int32)
      r_pred = y_pred[start + idx, 0]
      loss += (y[start, 0] * (avg_true_label - p) + p) * instance_loss(
          tf.convert_to_tensor([1.0]), tf.convert_to_tensor([r_pred])
      ) + (y[start, 0] * (p - avg_true_label) + (1 - p)) * instance_loss(
          tf.convert_to_tensor([0.0]), tf.convert_to_tensor([r_pred])
      )
      start = start + bag_size
  return tf.convert_to_tensor(loss)


def ot_llp_loss(
    batch_size,
    instance_loss,
    y,
    y_pred,
):
  """OT without ER loss."""
  start = 0
  loss = 0.0
  for _ in range(batch_size):
    bag_size = int(y[start, 0])
    sum_true_label = int(y[start + 1, 0])
    preds = y_pred[start : start + bag_size, 0]
    if bag_size != 0:
      if sum_true_label == 0:
        pseudo_labels = tf.zeros(tf.shape(preds), dtype=tf.float32)
      else:
        values, _ = tf.math.top_k(preds, k=sum_true_label)
        pseudo_labels = tf.cast(preds >= values[-1], tf.float32)
      loss += instance_loss(pseudo_labels, preds)
      start = start + bag_size
  return tf.convert_to_tensor(loss)


def sim_loss(repl, x, y_pred):
  """Similarity loss."""
  x_rep = repl(x)
  sigmoid_y_pred = tf.math.sigmoid(y_pred)
  diff_sq = tf.math.square(
      sigmoid_y_pred[:, tf.newaxis] - sigmoid_y_pred[tf.newaxis, :]
  )
  dot_prod_mat = tf.linalg.matmul(
      x_rep, x_rep, transpose_a=False, transpose_b=True
  )
  sq_norm = tf.linalg.diag_part(dot_prod_mat)
  coeff_mat = tf.math.exp(
      2.0 * dot_prod_mat - sq_norm[:, tf.newaxis] - sq_norm[tf.newaxis, :]
  )
  return tf.reduce_mean(coeff_mat * diff_sq)


def sim_llp_loss(
    sim_loss_size,
    bag_loss,
    lbd,
    repl,
    x,
    y,
    y_pred,
):
  """Sim-LLP loss."""
  indices = tf.random.shuffle(tf.range(tf.shape(y_pred)[0]))[:sim_loss_size]
  y_pred_rand = tf.gather(y_pred, indices, axis=0)
  x_rand = tf.gather(x, indices, axis=0)
  return sim_loss(repl, x_rand, y_pred_rand) + lbd * bag_loss(x, y, y_pred)


def dllp_loss_graph_regression(
    batch_size,
    instance_loss,
    x_categ,
    x_numer,
    y,
    y_pred,
):
  """DLLP loss for graph execution."""
  del x_categ, x_numer
  y_sg = tf.stop_gradient(y)
  loss = tf.zeros((), dtype=tf.float32)
  start = tf.zeros((), dtype=tf.int32)
  for _ in range(batch_size):
    bag_size = tf.cast(tf.gather(y_sg, start, axis=0)[0], dtype=tf.int32)
    avg_true_label = tf.gather(y_sg, start + 1, axis=0)[0]
    avg_pred_label = tf.reduce_mean(
        tf.gather(y_pred, tf.range(start, start + bag_size), axis=0)
    )
    loss += instance_loss(
        tf.convert_to_tensor([avg_true_label]),
        tf.convert_to_tensor([avg_pred_label]),
    )
    start += bag_size
  return loss / batch_size


def genbags_loss_regression(
    mean,
    covariance_mat,
    batch_size,
    block_size,
    num_gen_bags_per_block,
    x_categ,
    x_numer,
    y,
    y_pred,
):
  """Genbags loss."""
  del x_categ, x_numer
  loss = tf.zeros((), dtype=tf.float32)
  start = tf.zeros((), dtype=tf.int32)
  for _ in range(batch_size // block_size):
    wts = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=covariance_mat
    ).sample(num_gen_bags_per_block)
    list_of_diffs = []
    for _ in range(block_size):
      bag_size = tf.cast(tf.gather(y, start, axis=0)[0], dtype=tf.int32)
      avg_true_label = tf.gather(y, start + 1, axis=0)[0]
      avg_pred_label = tf.reduce_mean(
          tf.gather(y_pred, tf.range(start, start + bag_size), axis=0)
      )
      list_of_diffs.append(avg_pred_label - avg_true_label)
      start = start + bag_size
    loss = loss + tf.square(
        tf.norm(
            tf.linalg.matvec(wts, tf.convert_to_tensor(list_of_diffs)), ord=2
        )
    )
  return loss


def sim_loss_regression(
    repl,
    x_categ,
    x_numer,
    y_pred,
):
  """Similarity loss."""
  x_rep = repl.penultimate_rep(x_categ, x_numer)
  sigmoid_y_pred = tf.math.sigmoid(y_pred)
  diff_sq = tf.math.square(
      sigmoid_y_pred[:, tf.newaxis] - sigmoid_y_pred[tf.newaxis, :]
  )
  dot_prod_mat = tf.linalg.matmul(
      x_rep, x_rep, transpose_a=False, transpose_b=True
  )
  sq_norm = tf.linalg.diag_part(dot_prod_mat)
  coeff_mat = tf.math.exp(
      2.0 * dot_prod_mat - sq_norm[:, tf.newaxis] - sq_norm[tf.newaxis, :]
  )
  return tf.reduce_mean(coeff_mat * diff_sq)


def sim_llp_loss_regression(
    sim_loss_size,
    bag_loss,
    lbd,
    repl,
    x_categ,
    x_numer,
    y,
    y_pred,
):
  """Sim-LLP loss."""
  indices = tf.random.shuffle(tf.range(tf.shape(y_pred)[0]))[:sim_loss_size]
  y_pred_rand = tf.gather(y_pred, indices, axis=0)
  x_categ_rand = tf.gather(x_categ, indices, axis=0)
  x_numer_rand = tf.gather(x_numer, indices, axis=0)
  return sim_loss_regression(
      repl, x_categ_rand, x_numer_rand, y_pred_rand
  ) + lbd * bag_loss(x_categ, x_numer, y, y_pred)
