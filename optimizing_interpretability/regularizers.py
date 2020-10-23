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

"""Library containing Tensorflow implementations of gradient regularizers.

Each regularizer applies a penalty to model input gradients, i.e. the gradients
of some model output (e.g. logits, loss) with respect to the input.
"""

import re
import tensorflow as tf
from optimizing_interpretability import metrics_utils


def _compute_logit_gradients(logits, images, labels):
  """Computes input gradients at the logits layer of a model.

  Specifically, for each input image in a batch, computes the gradient of the
  logit for the true class of the input with respect to that input. This
  corresponds to the sensitivity (gradient) heatmap for that image.

  Note that, for images x and class logits y, we are interested in dy_i / dx_i
  for each (x_i, y_i) pair. tf.gradients(y, x) computes d(sum_i y_i) / dx_i for
  each x_i in x. The former and latter are equivalent as long as y_j is not a
  function of x_i for i != j, i.e. as long as the output for a given example
  depends only on that example and not on the rest of the batch. Standard image
  models may violate this constraint if using batch norm, but we can get around
  this problem by stopping gradient flow through the batch norm moment tensors.
  (Other violations must be handled explicitly by the user.)

  Args:
    logits: the unscaled model outputs. A Tensor of shape [batch_size,
      num_classes].
    images: the batch of inputs corresponding to the given outputs. A Tensor of
      shape [batch_size, height, width, channels].
    labels: one-hot encoded. A Tensor of shape [batch_size, num_classes].

  Returns:
    The input gradients, a Tensor of the same shape as images.
  """
  with tf.name_scope('computing_gradients'):
    class_logits = tf.reduce_sum(logits * labels, axis=1)
    # Collect all batch norm moment tensors in the graph, for stop_gradients.
    bn_ops = [
        x for x in tf.compat.v1.get_default_graph().get_operations()  # pylint: disable=g-complex-comprehension
        if re.search(
            '(batch_normalization|BatchNorm)[^/]*/moments/(mean|variance):',
            x.name)
    ]
    grads = tf.gradients(class_logits, images, stop_gradients=bn_ops)[0]

    return grads


def _normalize(batch):
  """Normalize a batch of images or gradient heatmaps to [0, 1]."""
  with tf.control_dependencies([tf.assert_equal(tf.rank(batch), 4)]):
    batch_min = tf.reduce_min(batch, axis=[1, 2, 3], keepdims=True)
    batch_max = tf.reduce_max(batch, axis=[1, 2, 3], keepdims=True)
    return (batch - batch_min) / (batch_max - batch_min)


def datagrad_regularizer(logits, images, labels):
  """L2 norm of input gradients of loss."""
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  grads = tf.gradients(loss, images)[0]
  return tf.reduce_mean(tf.square(grads))


# pylint: disable=unused-argument
def spectreg_regularizer(logits, images, labels):
  """L2 norm of input gradients of a random projection of logits."""
  logits_proj = logits * tf.random.normal(tf.shape(logits))
  grads = tf.gradients(logits_proj, images)[0]
  return tf.reduce_mean(tf.square(grads))
# pylint: enable=unused-argument


def l2_grad_regularizer(logits, images, labels):
  """L2 norm of the gradient heatmap."""
  grads = _compute_logit_gradients(logits, images, labels)
  return tf.reduce_mean(tf.square(grads))


def tv_grad_regularizer(logits, images, labels):
  """Total variation norm of the gradient heatmap."""
  grads = _compute_logit_gradients(logits, images, labels)
  reg_loss = tf.image.total_variation(grads)
  return tf.reduce_mean(reg_loss)


def tv_abs_grad_regularizer(logits, images, labels):
  """TV norm of the abs gradient heatmap after reducing channels, scaling."""
  grads = _compute_logit_gradients(logits, images, labels)
  grad_map = tf.reduce_sum(tf.abs(grads), -1, keep_dims=True)
  _, grad_map_var = tf.nn.moments(images, (1, 2), keep_dims=True)
  grad_map_standardized = grad_map / tf.sqrt(grad_map_var)
  return tf.reduce_mean(tf.image.total_variation(grad_map_standardized))


def tv_abs_unscaled_grad_regularizer(logits, images, labels):
  """TV norm of the abs gradient heatmap after reducing channels."""
  grads = _compute_logit_gradients(logits, images, labels)
  grad_map = tf.reduce_sum(tf.abs(grads), -1, keep_dims=True)
  return tf.reduce_mean(tf.image.total_variation(grad_map))


def mse_grad_regularizer(logits, images, labels):
  """Mean squared error between the gradient heatmap and image."""
  grads = _compute_logit_gradients(logits, images, labels)
  grads, images = _normalize(grads), _normalize(images)
  reg_loss = tf.squared_difference(grads, images)
  return tf.reduce_mean(reg_loss)


def cor_grad_regularizer(logits, images, labels):
  """Inverse correlation between the gradient heatmap and image. MSE variant."""
  grads = _compute_logit_gradients(logits, images, labels)
  image_means, image_vars = tf.nn.moments(images, (1, 2), keep_dims=True)
  grad_means, grad_vars = tf.nn.moments(grads, (1, 2), keep_dims=True)
  images_standardized = (images - image_means) / tf.sqrt(image_vars)
  grads_standardized = (grads - grad_means) / tf.sqrt(grad_vars)
  cor = tf.reduce_mean(images_standardized * grads_standardized, (1, 2))
  return 1 - tf.reduce_mean(tf.square(cor))


def graddiff_grad_regularizer(logits, images, labels):
  """Image gradient difference loss between the gradient heatmap and image."""
  grads = _compute_logit_gradients(logits, images, labels)
  grads, images = _normalize(grads), _normalize(images)
  reg_loss = tf.sqrt(metrics_utils.GradientDifferenceLoss(grads, images))
  return tf.reduce_mean(reg_loss)


def sobel_edges_grad_regularizer(logits, images, labels):
  """Sobel edge map loss between the gradient heatmap and image."""
  grads = _compute_logit_gradients(logits, images, labels)
  grads, images = _normalize(grads), _normalize(images)
  reg_loss = metrics_utils.SobelEdgeLoss(img1=grads, img2=images)
  return tf.reduce_mean(reg_loss)


def psnr_grad_regularizer(logits, images, labels):
  """Reciprocal peak signal-to-noise ratio for gradient heatmap and image."""
  grads = _compute_logit_gradients(logits, images, labels)
  grads, images = _normalize(grads), _normalize(images)
  reg_loss = tf.reciprocal(metrics_utils.PSNR(grads, images, max_val=1.))
  return tf.reduce_mean(reg_loss)


def ssim_unfiltered_grad_regularizer(logits, images, labels):
  """SSIM variant using a moving average, rather than Gaussian, filter."""
  grads = _compute_logit_gradients(logits, images, labels)
  grads, images = _normalize(grads), _normalize(images)
  reg_loss = -metrics_utils.SSIMWithoutFilter(grads, images, max_val=1.)
  return tf.reduce_mean(reg_loss)


REGULARIZERS = {
    'datagrad': datagrad_regularizer,
    'spectreg': spectreg_regularizer,
    'l2': l2_grad_regularizer,
    'tv': tv_grad_regularizer,
    'mse': mse_grad_regularizer,
    'grad_diff': graddiff_grad_regularizer,
    'sobel_edges': sobel_edges_grad_regularizer,
    'psnr': psnr_grad_regularizer,
    'ssim_unfiltered': ssim_unfiltered_grad_regularizer,
    'cor': cor_grad_regularizer,
    'tv_abs': tv_abs_grad_regularizer,
    'tv_abs_unscaled': tv_abs_unscaled_grad_regularizer
}


def compute_reg_loss(regularizer, logits, images, labels):
  """Computes the specified regularization loss.

  Args:
    regularizer: string name of the penalty to apply.
    logits: the unscaled model outputs. A Tensor of shape [batch_size,
      num_classes].
    images: the batch of inputs corresponding to the given outputs. A Tensor of
      shape [batch_size, height, width, channels].
    labels: one-hot encoded. A Tensor of shape [batch_size, num_classes].

  Returns:
    The regularization penalty, a scalar Tensor.

  Raises:
    KeyError: if regularizer is not among the available penalties.
  """
  if regularizer not in REGULARIZERS:
    raise KeyError('Regularizer not available.')
  with tf.name_scope(regularizer):
    reg_fn = REGULARIZERS[regularizer]
    return reg_fn(logits, images, labels)
