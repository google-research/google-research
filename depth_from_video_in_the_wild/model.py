# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A model for learning depth, egomotion and object 3D-motion field.

The method is described in https://arxiv.org/abs/1904.04998
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
from depth_from_video_in_the_wild import consistency_losses
from depth_from_video_in_the_wild import depth_prediction_net
from depth_from_video_in_the_wild import motion_prediction_net
from depth_from_video_in_the_wild import randomized_layer_normalization
from depth_from_video_in_the_wild import reader
from depth_from_video_in_the_wild import transform_depth_map
from depth_from_video_in_the_wild import transform_utils
from tensorflow.contrib import slim as contrib_slim


gfile = tf.gfile
slim = contrib_slim
# Number of subsequent frames per training sample. It is set to 3 for mainly
# legacy reasons: The training loss itself only involves two adjacent images at
# a time.
SEQ_LENGTH = 3
LAYER_NORM_NOISE_RAMPUP_STEPS = 10000
MIN_OBJECT_AREA = 20
DEPTH_SCOPE = 'depth_prediction'


class Model(object):
  """A model for running training and inference.

  Based on the Struct2Depth code:
  https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py
  """

  # pylint: disable=unused-argument
  # (we grab the arguments using locals())
  def __init__(self,
               data_dir=None,
               file_extension='png',
               is_training=True,
               learning_rate=1e-4,
               beta1=0.9,
               reconstr_weight=0.85,
               smooth_weight=1e-2,
               ssim_weight=3.0,
               batch_size=4,
               img_height=128,
               img_width=416,
               imagenet_norm=True,
               weight_reg=1e-2,
               random_scale_crop=False,
               random_color=True,
               shuffle=True,
               input_file='train',
               depth_consistency_loss_weight=1e-2,
               queue_size=2000,
               motion_smoothing_weight=1e-3,
               rotation_consistency_weight=1e-3,
               translation_consistency_weight=1e-2,
               foreground_dilation=8,
               learn_intrinsics=True,
               boxify=True):
    args = locals()
    for k in sorted(args):
      self.__dict__[k] = args[k]
      logging.info('%s: %s', k, str(args[k]))

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if self.is_training:
      self._build_train_graph()
    else:
      self._build_depth_test_graph()
      self._build_egomotion_test_graph()
  # pylint: enable=unused-argument

  def export(self, name, tensor):
    """Maintains a set of tensors to evaluate for debugging.

    Args:
      name: A string, a name for saving / printing the results.
      tensor: A tensor to be evaluated for debugging.
    """
    if not hasattr(self, 'exports'):
      self.exports = {}
    if not isinstance(tensor, tf.Tensor):
      raise ValueError(
          'Only tensors can be exported, not %s (for %s)' % (str(tensor), name))
    self.exports[name] = tensor

  def _build_train_graph(self):
    """Build a training graph and savers."""
    self._build_loss()
    self.saver = tf.train.Saver()
    # Create a saver for initializing resnet18 weights from imagenet.
    vars_to_restore = [
        v for v in tf.trainable_variables()
        if v.op.name.startswith(DEPTH_SCOPE + '/conv')
    ]
    vars_to_restore = {
        v.op.name[len(DEPTH_SCOPE) + 1:]: v for v in vars_to_restore
    }
    self.imagenet_init_restorer = tf.train.Saver(vars_to_restore)
    self._build_train_op()
    self._build_summaries()

  def _build_loss(self):
    """Builds the loss tensor, to be minimized by the optimizer."""
    self.reader = reader.DataReader(
        self.data_dir,
        self.batch_size,
        self.img_height,
        self.img_width,
        SEQ_LENGTH,
        1,  # num_scales
        self.file_extension,
        self.random_scale_crop,
        reader.FLIP_RANDOM,
        self.random_color,
        self.imagenet_norm,
        self.shuffle,
        self.input_file,
        queue_size=self.queue_size)

    (self.image_stack, self.image_stack_norm, self.seg_stack,
     self.intrinsic_mat, _) = self.reader.read_data()
    if self.learn_intrinsics:
      self.intrinsic_mat = None
    if self.intrinsic_mat is None and not self.learn_intrinsics:
      raise RuntimeError('Could not read intrinsic matrix. Turn '
                         'learn_intrinsics on to learn it instead of loading '
                         'it.')
    self.export('self.image_stack', self.image_stack)

    object_masks = []
    for i in range(self.batch_size):
      object_ids = tf.unique(tf.reshape(self.seg_stack[i], [-1]))[0]
      object_masks_i = []
      for j in range(SEQ_LENGTH):
        current_seg = self.seg_stack[i, :, :, j * 3]  # (H, W)
        def process_obj_mask(obj_id):
          """Create a mask for obj_id, skipping the background mask."""
          mask = tf.logical_and(
              tf.equal(current_seg, obj_id),  # pylint: disable=cell-var-from-loop
              tf.not_equal(tf.cast(0, tf.uint8), obj_id))
          # Leave out vert small masks, that are most often errors.
          size = tf.reduce_sum(tf.to_int32(mask))
          mask = tf.logical_and(mask, tf.greater(size, MIN_OBJECT_AREA))
          if not self.boxify:
            return mask
          # Complete the mask to its bounding box.
          binary_obj_masks_y = tf.reduce_any(mask, axis=1, keepdims=True)
          binary_obj_masks_x = tf.reduce_any(mask, axis=0, keepdims=True)
          return tf.logical_and(binary_obj_masks_y, binary_obj_masks_x)

        object_mask = tf.map_fn(  # (N, H, W)
            process_obj_mask, object_ids, dtype=tf.bool)
        object_mask = tf.reduce_any(object_mask, axis=0)
        object_masks_i.append(object_mask)
      object_masks.append(tf.stack(object_masks_i, axis=-1))

    self.seg_stack = tf.to_float(tf.stack(object_masks, axis=0))
    tf.summary.image('Masks', self.seg_stack)

    with tf.variable_scope(DEPTH_SCOPE):
      # Organized by ...[i][scale].  Note that the order is flipped in
      # variables in build_loss() below.
      self.disp = {}
      self.depth = {}

      # Parabolic rampup of he noise over LAYER_NORM_NOISE_RAMPUP_STEPS steps.
      # We stop at 0.5 because this is the value above which the multiplicative
      # noise we use can become negative. Further experimentation is needed to
      # find if non-negativity is indeed needed.
      noise_stddev = 0.5 * tf.square(
          tf.minimum(
              tf.to_float(self.global_step) /
              float(LAYER_NORM_NOISE_RAMPUP_STEPS), 1.0))

      def _normalizer_fn(x, is_train, name='bn'):
        return randomized_layer_normalization.normalize(
            x, is_train=is_train, name=name, stddev=noise_stddev)

      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(SEQ_LENGTH):
          image = self.image_stack_norm[:, :, :, 3 * i:3 * (i + 1)]
          self.depth[i] = depth_prediction_net.depth_prediction_resnet18unet(
              image, True, self.weight_reg, _normalizer_fn)
          self.disp[i] = 1.0 / self.depth[i]

    with tf.name_scope('compute_loss'):
      self.reconstr_loss = 0
      self.smooth_loss = 0
      self.ssim_loss = 0
      self.depth_consistency_loss = 0

      # Smoothness.
      if self.smooth_weight > 0:
        for i in range(SEQ_LENGTH):
          disp_smoothing = self.disp[i]
          # Perform depth normalization, dividing by the mean.
          mean_disp = tf.reduce_mean(
              disp_smoothing, axis=[1, 2, 3], keep_dims=True)
          disp_input = disp_smoothing / mean_disp
          self.smooth_loss += _depth_smoothness(
              disp_input, self.image_stack[:, :, :, 3 * i:3 * (i + 1)])

      self.rot_loss = 0.0
      self.trans_loss = 0.0

      def add_result_to_loss_and_summaries(endpoints, i, j):
        tf.summary.image(
            'valid_mask%d%d' % (i, j),
            tf.expand_dims(endpoints['depth_proximity_weight'], -1))

        self.depth_consistency_loss += endpoints['depth_error']
        self.reconstr_loss += endpoints['rgb_error']
        self.ssim_loss += 0.5 * endpoints['ssim_error']
        self.rot_loss += endpoints['rotation_error']
        self.trans_loss += endpoints['translation_error']

      self.motion_smoothing = 0.0
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(SEQ_LENGTH - 1):
          j = i + 1
          depth_i = self.depth[i][:, :, :, 0]
          depth_j = self.depth[j][:, :, :, 0]
          image_j = self.image_stack[:, :, :, 3 * j:3 * (j + 1)]
          image_i = self.image_stack[:, :, :, i*3:(i+1)*3]
          # We select a pair of consecutive images (and their respective
          # predicted depth maps). Now we have the network predict a motion
          # field that connects the two. We feed the pair of images into the
          # network, once in forward order and then in reverse order. The
          # results are fed into the loss calculation. The following losses are
          # calculated:
          # - RGB and SSIM photometric consistency.
          # - Cycle consistency of rotations and translations for every pixel.
          # - L1 smoothness of the disparity and the motion field.
          # - Depth consistency
          rot, trans, trans_res, mat = motion_prediction_net.motion_field_net(
              images=tf.concat([image_i, image_j], axis=-1),
              weight_reg=self.weight_reg)
          inv_rot, inv_trans, inv_trans_res, inv_mat = (
              motion_prediction_net.motion_field_net(
                  images=tf.concat([image_j, image_i], axis=-1),
                  weight_reg=self.weight_reg))

          if self.learn_intrinsics:
            intrinsic_mat = 0.5 * (mat + inv_mat)
          else:
            intrinsic_mat = self.intrinsic_mat[:, 0, :, :]

          def dilate(x):
            # Dilation by n pixels is roughtly max pooling by 2 * n + 1.
            p = self.foreground_dilation * 2 + 1
            return tf.nn.max_pool(x, [1, p, p, 1], [1]*4, 'SAME')

          trans += trans_res * dilate(self.seg_stack[:, :, :, j:j + 1])
          inv_trans += inv_trans_res * dilate(self.seg_stack[:, :, :, i:i + 1])

          tf.summary.image('trans%d%d' % (i, i+1), trans)
          tf.summary.image('trans%d%d' % (i+1, i), inv_trans)

          tf.summary.image('trans_res%d%d' % (i+1, i), inv_trans_res)
          tf.summary.image('trans_res%d%d' % (i, i+1), trans_res)

          self.motion_smoothing += _smoothness(trans)
          self.motion_smoothing += _smoothness(inv_trans)
          tf.summary.scalar(
              'trans_stdev',
              tf.sqrt(0.5 *
                      tf.reduce_mean(tf.square(trans) + tf.square(inv_trans))))

          transformed_depth_j = transform_depth_map.using_motion_vector(
              depth_j, trans, rot, intrinsic_mat)

          add_result_to_loss_and_summaries(
              consistency_losses.rgbd_and_motion_consistency_loss(
                  transformed_depth_j, image_j, depth_i, image_i, rot,
                  trans, inv_rot, inv_trans), i, j)

          transformed_depth_i = transform_depth_map.using_motion_vector(
              depth_i, inv_trans, inv_rot, intrinsic_mat)

          add_result_to_loss_and_summaries(
              consistency_losses.rgbd_and_motion_consistency_loss(
                  transformed_depth_i, image_i, depth_j, image_j, inv_rot,
                  inv_trans, rot, trans), j, i)

      # Build the total loss as composed of L1 reconstruction, SSIM, smoothing
      # and object size constraint loss as appropriate.
      self.reconstr_loss *= self.reconstr_weight
      self.export('self.reconstr_loss', self.reconstr_loss)
      self.total_loss = self.reconstr_loss
      if self.smooth_weight > 0:
        self.smooth_loss *= self.smooth_weight
        self.total_loss += self.smooth_loss
        self.export('self.smooth_loss', self.smooth_loss)
      if self.ssim_weight > 0:
        self.ssim_loss *= self.ssim_weight
        self.total_loss += self.ssim_loss
        self.export('self.ssim_loss', self.ssim_loss)

      if self.motion_smoothing_weight > 0:
        self.motion_smoothing *= self.motion_smoothing_weight
        self.total_loss += self.motion_smoothing
        self.export('self.motion_sm_loss', self.motion_smoothing)

      if self.depth_consistency_loss_weight:
        self.depth_consistency_loss *= self.depth_consistency_loss_weight
        self.total_loss += self.depth_consistency_loss
        self.export('self.depth_consistency_loss', self.depth_consistency_loss)

      self.rot_loss *= self.rotation_consistency_weight
      self.trans_loss *= self.translation_consistency_weight
      self.export('rot_loss', self.rot_loss)
      self.export('trans_loss', self.trans_loss)

      self.total_loss += self.rot_loss
      self.total_loss += self.trans_loss

      self.export('self.total_loss', self.total_loss)

  def _build_train_op(self):
    with tf.name_scope('train_op'):
      optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
      self.train_op = slim.learning.create_train_op(self.total_loss, optim,
                                                    clip_gradient_norm=10.0)

  def _build_summaries(self):
    """Adds scalar and image summaries for TensorBoard."""
    tf.summary.scalar('total_loss', self.total_loss)
    tf.summary.scalar('reconstr_loss', self.reconstr_loss)
    if self.smooth_weight > 0:
      tf.summary.scalar('smooth_loss', self.smooth_loss)
    if self.ssim_weight > 0:
      tf.summary.scalar('ssim_loss', self.ssim_loss)
    if self.motion_smoothing_weight > 0:
      tf.summary.scalar('motion_smoothing', self.motion_smoothing)

    tf.summary.scalar('rotation_consistency_loss', self.rot_loss)
    tf.summary.scalar('translation_consistency_loss', self.trans_loss)

    if self.depth_consistency_loss_weight > 0:
      tf.summary.scalar('depth_consistency_loss', self.depth_consistency_loss)

    for i in range(SEQ_LENGTH):
      tf.summary.image('image%d' % i,
                       self.image_stack[:, :, :, 3 * i:3 * (i + 1)])
      if i in self.depth:
        tf.summary.histogram('depth%d' % i, self.depth[i])
        tf.summary.histogram('disp%d' % i, self.disp[i])
        tf.summary.image('disparity%d' % i, self.disp[i])

  def _build_depth_test_graph(self):
    """Builds depth model reading from placeholders."""
    with tf.variable_scope(DEPTH_SCOPE, reuse=tf.AUTO_REUSE):
      input_image = tf.placeholder(
          tf.float32, [self.batch_size, self.img_height, self.img_width, 3],
          name='raw_input')
      if self.imagenet_norm:
        input_image = (input_image - reader.IMAGENET_MEAN) / reader.IMAGENET_SD

      def _normalizer_fn(x, is_train, name='bn'):
        return randomized_layer_normalization.normalize(
            x, is_train, name, None)

      self.est_depth1 = depth_prediction_net.depth_prediction_resnet18unet(
          images=input_image, is_training=False, normalizer_fn=_normalizer_fn)
      self.est_depth2 = tf.image.flip_left_right(
          depth_prediction_net.depth_prediction_resnet18unet(
              images=tf.image.flip_left_right(input_image),
              is_training=False,
              normalizer_fn=_normalizer_fn))
      self.est_depth = tf.minimum(self.est_depth1, self.est_depth2)

    self.input_image = input_image

  def inference_depth(self, inputs, sess):
    return sess.run(self.est_depth, feed_dict={self.input_image: inputs})

  def _build_egomotion_test_graph(self):
    """Builds graph for inference of egomotion given two images."""
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self._image1 = tf.placeholder(
          tf.float32, [self.batch_size, self.img_height, self.img_width, 3],
          name='image1')
      self._image2 = tf.placeholder(
          tf.float32, [self.batch_size, self.img_height, self.img_width, 3],
          name='image2')
      # The "compute_loss" scope is needed for the checkpoint to load properly.
      with tf.name_scope('compute_loss'):
        rot, trans, _, _ = motion_prediction_net.motion_field_net(
            images=tf.concat([self._image1, self._image2], axis=-1))
        inv_rot, inv_trans, _, _ = (
            motion_prediction_net.motion_field_net(
                images=tf.concat([self._image2, self._image1], axis=-1)))

      rot = transform_utils.matrix_from_angles(rot)
      inv_rot = transform_utils.matrix_from_angles(inv_rot)
      trans = tf.squeeze(trans, axis=(1, 2))
      inv_trans = tf.squeeze(inv_trans, axis=(1, 2))

      # rot and inv_rot should be the inverses on of the other, but in reality
      # they slightly differ. Averaging rot and inv(inv_rot) gives a better
      # estimator for the rotation. Similarly, trans and rot*inv_trans should
      # be the negatives one of the other, so we average rot*inv_trans and trans
      # to get a better estimator. TODO(gariel): Check if there's an estimator
      # with less variance.
      self.rot = 0.5 * (tf.linalg.inv(inv_rot) + rot)
      self.trans = 0.5 * (-tf.squeeze(
          tf.matmul(self.rot, tf.expand_dims(inv_trans, -1)), axis=-1) + trans)

  def inference_egomotion(self, image1, image2, sess):
    return sess.run([self.rot, self.trans],
                    feed_dict={
                        self._image1: image1,
                        self._image2: image2
                    })


def _gradient_x(img):
  return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
  return img[:, :-1, :, :] - img[:, 1:, :, :]


def _depth_smoothness(depth, img):
  """Computes image-aware depth smoothness loss."""
  depth_dx = _gradient_x(depth)
  depth_dy = _gradient_y(depth)
  image_dx = _gradient_x(img)
  image_dy = _gradient_y(img)
  weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
  weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
  smoothness_x = depth_dx * weights_x
  smoothness_y = depth_dy * weights_y
  return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))


def _smoothness(motion_map):
  norm = tf.reduce_mean(
      tf.square(motion_map), axis=[1, 2, 3], keep_dims=True) * 3.0
  motion_map /= tf.sqrt(norm + 1e-12)
  return _smoothness_helper(motion_map)


def _smoothness_helper(motion_map):
  """Calculates L1 (total variation) smoothness loss of a tensor.

  Args:
    motion_map: A tensor to be smoothed, of shape [B, H, W, C].

  Returns:
    A scalar tf.Tensor, The total variation loss.
  """
  # We roll in order to impose continuity across the boundary. The motivation is
  # that there is some ambiguity between rotation and spatial gradients of
  # translation maps. We would like to discourage spatial gradients of the
  # translation field, and to absorb sich gradients into the rotation as much as
  # possible. This is why we impose continuity across the spatial boundary.
  motion_map_dx = motion_map - tf.roll(motion_map, 1, 1)
  motion_map_dy = motion_map - tf.roll(motion_map, 1, 2)
  sm_loss = tf.sqrt(1e-24 + tf.square(motion_map_dx) + tf.square(motion_map_dy))
  tf.summary.image('motion_sm', sm_loss)
  return tf.reduce_mean(sm_loss)
