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

"""Loss and metric definitions for KeyPose models."""

import numpy as np
import tensorflow as tf

from keypose import nets

# num_targs = inp.MAX_TARGET_FRAMES
num_targs = 5

# Reordering based on symmetry.


def make_order(sym, num_kp):
  """Returns all rotations of the <sym> keypoints."""
  rots = np.array([sym[-i:] + sym[:-i] for i in range(len(sym))])
  rot_list = np.array([range(num_kp)] * len(sym))
  for i, rot in enumerate(rot_list):
    rot[rots[i]] = rot_list[0][rots[0]]
  return rot_list


def reorder(tensor, order):
  """Re-orders a tensor along the num_kp dimension.

  Args:
    tensor: has shape [batch, num_kp, ...]
    order: permutation of keypoints.

  Returns:
    shape [batch, order, num_kp, ...]
  """
  return tf.stack(
      # order is list of lists, so pylint: disable=not-an-iterable
      [tf.stack([tensor[:, x, ...] for x in ord], axis=1) for ord in order],
      axis=1)


def reduce_order(tensor, mask, mean=False):
  res = tf.multiply(tensor, mask)
  res = tf.reduce_sum(res, axis=1)
  if mean:
    return tf.reduce_mean(res)
  else:
    return res


# Loss functions.


def project(tmat, tvec, tvec_transpose=False):
  """Projects homogeneous 3D XYZ coordinates to image uvd coordinates, or vv.

  Args:
    tmat: has shape [[N,] batch_size, 4, 4].
    tvec: has shape [[N,] batch_size, 4, num_kp] or [batch_size, num_kp, 4].
    tvec_transpose: True if tvec is to be transposed before application.

  Returns:
    Has shape [[N,] batch_size, 4, num_kp].
  """
  tp = tf.matmul(tmat, tvec, transpose_b=tvec_transpose)
  # Using <3:4> instead of <3> preserves shape.
  tp = tp / (tp[..., 3:4, :] + 1.0e-10)
  return tp


def keypoint_loss_targets(uvd, keys_uvd, mparams):
  """Computes the supervised keypoint loss between computed and gt keypoints.

  Args:
    uvd: [batch, order, num_targs, 4, num_kp] Predicted set of keypoint uv's
      (pixels).
    keys_uvd: [batch, order, num_targs, 4, num_kp] The ground-truth set of uvdw
      coords.
    mparams: model parameters.

  Returns:
    Keypoint projection loss of size [batch, order].
  """
  print('uvd shape in klt [batch, order, num_targs, 4, num_kp]:', uvd.shape)
  print('keys_uvd shape in klt [batch, order, num_targs, 4, num_kp]:',
        keys_uvd.shape)
  keys_uvd = nets.to_norm(keys_uvd, mparams)
  uvd = nets.to_norm(uvd, mparams)

  wd = tf.square(uvd[..., :2, :] - keys_uvd[..., :2, :])
  wd = tf.reduce_sum(wd, axis=[-1, -2])  # uv dist [batch, order, num_targs]
  print('wd shape in klt [batch, order, num_targs]:', wd.shape)
  wd = tf.reduce_mean(wd, axis=[-1])  # [batch, order]
  return wd


# Compute the reprojection error on the target frames.
def keypose_loss_proj(uvdw_pos, labels, mparams, num_order):
  """Compute the reprojection error on the target frames.

  Args:
    uvdw_pos: predicted uvd, always positive.
    labels: sample labels.
    mparams: model parameters.
    num_order: number of order permutations.

  Returns:
    Scalar loss.
  """
  num_kp = mparams.num_kp
  to_world = labels['to_world_L']  # [batch, 4, 4]
  to_world_order = tf.stack(
      [to_world] * num_order, axis=1)  # [batch, order, 4, 4]
  to_world_order = tf.ensure_shape(
      to_world_order, [None, num_order, 4, 4], name='to_world_order')
  world_coords = project(to_world_order, uvdw_pos,
                         True)  # [batch, order, 4, num_kp]
  world_coords = tf.ensure_shape(
      world_coords, [None, num_order, 4, num_kp], name='world_coords')
  print('world_coords shape [batch, order, 4, num_kp]:', world_coords.shape)

  # Target transform and keypoints.
  # [batch, num_targs, 4, 4] for transforms
  # [batch, num_targs, 4, num_kp] for keypoints (after transpose)
  targets_to_uvd = labels['targets_to_uvd_L']
  targets_keys_uvd = tf.transpose(labels['targets_keys_uvd_L'], [0, 1, 3, 2])
  targets_keys_uvd_order = tf.stack([targets_keys_uvd] * num_order, axis=1)
  print('Model fn targets_to_uvd shape [batch, num_targs, 4, 4]:',
        targets_to_uvd.shape)
  print(
      'Model fn targets_keys_uvd_order shape [batch, order, num_targs, 4, '
      'num_kp]:', targets_keys_uvd_order.shape)

  # [batch, order, num_targs, 4, num_kp]
  proj_uvds = project(
      tf.stack([targets_to_uvd] * num_order, axis=1),
      tf.stack([world_coords] * num_targs, axis=2))
  proj_uvds = tf.ensure_shape(
      proj_uvds, [None, num_order, 5, 4, num_kp], name='proj_uvds')
  print('proj_uvds shape [batch, order, num_targs, 4, num_kp]:',
        proj_uvds.shape)
  loss_proj = keypoint_loss_targets(proj_uvds, targets_keys_uvd_order, mparams)
  loss_proj = tf.ensure_shape(loss_proj, [None, num_order], name='loss_proj')
  print('loss_proj shape [batch, order]:', loss_proj.shape)
  return loss_proj


# Keypoint loss function, direct comparison of u,v,d.
# Compares them in normalized image coordinates.  For some reason this
# seems to work better than pixels.
# uvdw_order: [batch, order, num_kp, 4]
# keys_uvd_order: [batch, order, num_kp, 4]
# Returns: [batch, order]
def keypose_loss_kp(uvdw_order, keys_uvd_order, mparams):
  uvdw_order = nets.to_norm_vec(uvdw_order, mparams)
  keys_uvd_order = nets.to_norm_vec(keys_uvd_order, mparams)
  ret = tf.reduce_sum(
      tf.square(uvdw_order[..., :3] - keys_uvd_order[..., :3]),
      axis=[-1, -2])  # [batch, order]
  return ret


# Probability coherence loss.
def keypose_loss_prob(prob_order, prob_label_order):
  ret = tf.reduce_sum(
      prob_order * prob_label_order, axis=[-1, -2])  # [batch, order]
  return tf.reduce_mean(ret, axis=-1)


# Adjust the gain of loss_proj; return in [0,1].
def adjust_proj_factor(step, loss_step, minf=0.0):
  if loss_step[1] == 0:
    return 1.0
  step = tf.cast(step, tf.float32)
  return tf.maximum(
      minf,
      tf.minimum((step - loss_step[0]) /
                 tf.cast(loss_step[1] - loss_step[0], tf.float32), 1.0))


# Custom loss function for the Keras model in Estimator
# Args are:
#  Dict of tensors for labels, with keys_uvd, offsets.
#  Tensor of raw uvd values for preds, [batch, num_kp, 3],
#    order is [u,v,d].
# Note that the loss is batched.
# This loss only works with tf Estimator, not keras models.
@tf.function
def keypose_loss(labels, preds, step, mparams, do_print=True):
  """Custom loss function for the Keras model in Estimator.

  Note that the loss is batched.
  This loss only works with tf Estimator, not keras models.

  Args:
    labels: dict of tensors for labels, with keys_uvd, offsets.
    preds: tensor of raw uvd values for preds, [batch, num_kp, 3], order is
      [u,v,d].
    step: training step.
    mparams: model training parameters.
    do_print: True to print loss values at every step.

  Returns:
    Scalar loss.
  """
  num_kp = mparams.num_kp
  sym = mparams.sym
  order = make_order(sym, num_kp)
  num_order = len(order)

  uvdw = preds['uvdw']
  uvdw_pos = preds['uvdw_pos']
  uv_pix_raw = preds['uv_pix_raw']
  prob = preds['prob']  # [batch, num_kp, resy, resx]
  xyzw = tf.transpose(preds['xyzw'], [0, 2, 1])  # [batch, num_kp, 4]

  uvdw_order = reorder(uvdw, order)  # [batch, order, num_kp, 4]
  print('uvdw_order shape:', uvdw_order.shape)
  uvdw_pos_order = reorder(uvdw_pos, order)  # [batch, order, num_kp, 4]
  xyzw_order = reorder(xyzw, order)  # [batch, order, 4, num_kp]
  print('xyzw_order shape:', xyzw_order.shape)
  prob_order = reorder(prob, order)  # [ batch, order, num_kp, resy, resx]
  print('prob_order shape:', prob_order.shape)

  keys_uvd = labels['keys_uvd_L']  # [batch, num_kp, 4]
  # [batch, order, num_kp, 4]
  keys_uvd_order = tf.stack([keys_uvd] * num_order, axis=1)

  loss_kp = keypose_loss_kp(uvdw_order, keys_uvd_order,
                            mparams)  # [batch, order]
  loss_kp.set_shape([None, num_order])
  # [batch, order]
  loss_proj = keypose_loss_proj(uvdw_pos_order, labels, mparams, num_order)
  loss_proj.set_shape([None, num_order])
  loss_proj_adj = adjust_proj_factor(step, mparams.loss_proj_step)

  prob_label = labels['prob_label']  # [batch, num_kp, resy, resx]
  # [batch, order, num_kp, resy, resx]
  prob_label_order = tf.stack([prob_label] * num_order, axis=1)
  loss_prob = keypose_loss_prob(prob_order, prob_label_order)

  loss_order = (
      mparams.loss_kp * loss_kp +
      mparams.loss_proj * loss_proj_adj * loss_proj +
      mparams.loss_prob * loss_prob)

  print('loss_order shape [batch, order]:', loss_order.shape)
  loss = tf.reduce_min(loss_order, axis=1, keepdims=True)  # shape [batch, 1]
  print('loss shape [batch, 1]:', loss.shape)

  loss_mask = tf.cast(tf.equal(loss, loss_order), tf.float32)  # [batch, order]
  loss_mask3 = tf.expand_dims(loss_mask, -1)  # [batch, order, 1]
  loss_mask4 = tf.expand_dims(loss_mask3, -1)  # [batch, order, 1, 1]
  print('loss_mask shape [batch, order]:', loss_mask.shape)

  loss = tf.reduce_mean(loss)  # Scalar, reduction over batch.
  loss_kp = reduce_order(loss_kp, loss_mask, mean=True)
  loss_proj = reduce_order(loss_proj, loss_mask, mean=True)
  loss_prob = reduce_order(loss_prob, loss_mask, mean=True)

  uvdw = reduce_order(uvdw_order, loss_mask4)  # [batch, num_kp, 4]
  xyzw = reduce_order(xyzw_order, loss_mask4)  # [batch, num_kp, 4]
  print('xyzw shape:', xyzw.shape)

  if do_print:
    tf.print(
        '  ',
        step,
        'Keypose loss:',
        loss,
        mparams.loss_kp * loss_kp,
        mparams.loss_proj * loss_proj_adj * loss_proj,
        mparams.loss_prob * loss_prob,
        '  ',
        loss_proj_adj,
        uv_pix_raw[0, 0, :3],
        uvdw_pos[0, 0, :3],
        keys_uvd[0, 0, :3],
        summarize=-1)
  return loss, uvdw, xyzw


#
# Metrics and visualization.
#


def add_keypoints(img, uv, colors=None):
  """Add keypoint markers to an image, using draw_bounding_boxes.

  Args:
    img: [batch, vh, vw, 3]
    uv: [batch, num_kp, 2], in normalized coords [-1,1], xy order.
    colors: color palette for keypoints.

  Returns:
    tf images with drawn keypoints.
  """
  if colors is None:
    colors = tf.constant([[0.0, 1.0, 0.0, 1.0]])
  else:
    colors = tf.constant(colors)
  uv = uv[:, :, :2] * 0.5 + 0.5  # [-1,1] -> [0,1]
  keys_bb_ul = tf.stack([uv[:, :, 1], uv[:, :, 0]], axis=2)
  keys_bb_lr = keys_bb_ul + 3.0 / tf.cast(tf.shape(img)[1], dtype=tf.float32)
  keys_bb = tf.concat([keys_bb_ul, keys_bb_lr], axis=2)
  print('Bounding box shape:', keys_bb.shape)
  return tf.image.draw_bounding_boxes(img, tf.cast(keys_bb, dtype=tf.float32),
                                      colors)


def add_keypoints_uv(img, uv, colors=None):
  """Add keypoint markers to an image, using draw_bounding_boxes.

  Args:
    img: [batch, vh, vw, 3]
    uv: [batch, num_kp, 2], in image coords, xy order.
    colors: color palette for drawing keypoints.

  Returns:
    tf images with drawn keypoints.
  """
  resy = img.shape[1]
  resx = img.shape[2]
  uvx = uv[:, :, 0] / resx
  uvy = uv[:, :, 1] / resy
  uv = tf.stack([uvx, uvy], axis=2)
  return add_keypoints(img, (uv - 0.5) * 2.0, colors)


def uv_error(labels, uvdw, _):
  diff = labels['keys_uvd_L'][..., :2] - uvdw[..., :2]
  return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))  # [batch, num_kp]


def disp_error(labels, uvdw, _):
  return tf.abs(labels['keys_uvd_L'][..., 2] - uvdw[..., 2])  # [batch, num_kp]


def world_error(labels, xyzw):
  xyzw = tf.transpose(xyzw, [0, 2, 1])  # [batch, 4, num_kp]
  # [batch, 4, num_kp]
  gt_world_coords = project(labels['to_world_L'], labels['keys_uvd_L'], True)
  sub = xyzw[:, :3, :] - gt_world_coords[:, :3, :]
  wd = tf.square(sub)
  wd = tf.reduce_sum(wd, axis=[-2])  # [batch, num_kp] result.
  wd = tf.sqrt(wd)
  return wd  # [batch, num_kp]


def lt_2cm_error(labels, xyzw):
  err = world_error(labels, xyzw)
  lt = tf.less(err, 0.02)
  return (100.0 * tf.cast(tf.math.count_nonzero(lt, axis=[-1]), tf.float32) /
          tf.cast(tf.shape(err)[1], tf.float32))  # [batch]
