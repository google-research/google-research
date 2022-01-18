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

# Lint as: python3
"""Loss functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from saccader import utils
from saccader.visual_attention import saccader


def normalized_l2_loss(var_list):
  """Computes l2 regularization loss normalized by number of parameters."""
  trainable_variables = tf.trainable_variables()
  count = 0
  reg_loss = 0.
  for v in var_list:
    if ("bias" not in v.name and "Norm" not in v.name and
        v in trainable_variables):
      reg_loss += tf.nn.l2_loss(v)
      count += np.prod(v.shape.as_list())
  reg_loss = 2. * reg_loss / count
  return reg_loss


def reinforce_loss_continuous(classification_logits_t,
                              classification_labels_t,
                              locations_t,
                              mean_locations_t,
                              std_locations,
                              use_punishment=False):
  """Computes REINFORCE loss for continuous action spaces.

  Args:
    classification_logits_t: List of logits of the model at each time point.
    classification_labels_t: List of data labels at each time point.
    locations_t: List of location samples at each time point.
    mean_locations_t: List of mean location estimate at each time point.
    std_locations: STD of location sampling function.
    use_punishment: (Boolean) Reward {-1, 1} if true else {0, 1}.
  Returns:
    reinforce_loss: REINFORCE loss.
  """
  classification_logits = tf.concat(classification_logits_t, axis=0)
  classification_labels = tf.concat(classification_labels_t, axis=0)
  locations_t = tf.concat(locations_t, axis=0)
  mean_locations_t = tf.concat(mean_locations_t, axis=0)

  rewards = tf.cast(
      tf.equal(
          tf.argmax(classification_logits, axis=1,
                    output_type=classification_labels.dtype),
          classification_labels), dtype=tf.float32)  # Size (batch_size) each
  if use_punishment:
    rewards = 2. * rewards - 1.

  rewards = tf.stop_gradient(rewards)
  prob_l = tfp.distributions.Normal(loc=mean_locations_t, scale=std_locations)
  log_prob_l = tf.reduce_sum(prob_l.log_prob(locations_t), axis=-1)
  neg_advs = (rewards - tf.stop_gradient(tf.reduce_mean(rewards)))

  reinforce_loss = -tf.reduce_mean(neg_advs * log_prob_l)

  return reinforce_loss


def reinforce_loss_discrete(classification_logits_t,
                            classification_labels_t,
                            locations_logits_t,
                            locations_labels_t,
                            use_punishment=False):
  """Computes REINFORCE loss for contentious discrete action spaces.

  Args:
    classification_logits_t: List of classification logits at each time point.
    classification_labels_t: List of classification labels at each time point.
    locations_logits_t: List of location logits at each time point.
    locations_labels_t: List of location labels at each time point.
    use_punishment: (Boolean) Reward {-1, 1} if true else {0, 1}.

  Returns:
    reinforce_loss: REINFORCE loss.
  """
  classification_logits = tf.concat(classification_logits_t, axis=0)
  classification_labels = tf.concat(classification_labels_t, axis=0)
  locations_logits = tf.concat(locations_logits_t, axis=0)
  locations_labels = tf.concat(locations_labels_t, axis=0)
  rewards = tf.cast(
      tf.equal(
          tf.argmax(classification_logits, axis=1,
                    output_type=classification_labels.dtype),
          classification_labels), dtype=tf.float32)  # size (batch_size) each
  if use_punishment:
    # Rewards is \in {-1 and 1} instead of {0, 1}.
    rewards = 2. * rewards - 1.
  neg_advs = tf.stop_gradient(rewards - tf.reduce_mean(rewards))
  log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=locations_logits, labels=locations_labels)

  loss = -tf.reduce_mean(neg_advs * log_prob)
  return loss


def reconstruction_losses(images,
                          reconstructed_images,
                          locations_t,
                          mean_locations_t,
                          std_locations,
                          norm="l2"):
  """Computes image reconstruction losses.

  Args:
    images: 4D tensor of images with shape [batch, H, W, C].
    reconstructed_images: 4D tensor of reconstructed images with shape [batch,
      H, W, C].
    locations_t: List of location samples at each time point.
    mean_locations_t: List of mean location estimate at each time point.
    std_locations: STD of location sampling function.
    norm: Error norm. 'l2': for square error. 'l1': for L1 error.

  Returns:
    reconstruction_loss: Mean error in reconstruction.
    reinforce_reconstruction_loss: REINFORCE loss with reward equal to negative
      the reconstruction error.
  """
  num_times = len(locations_t)
  images = tf.layers.flatten(images)
  reconstructed_images = tf.layers.flatten(reconstructed_images)
  if norm == "l2":
    reconstruction_err = tf.reduce_mean(
        tf.math.squared_difference(images, reconstructed_images), axis=1)
  elif norm == "l1":
    reconstruction_err = tf.reduce_mean(
        tf.math.abs(images - reconstructed_images), axis=1)
  else:
    raise ValueError("Invalid norm")

  # Fixed reward for all times based on last time prediction.
  rewards = tf.tile([-reconstruction_err], [num_times, 1])
  rewards = tf.stop_gradient(rewards)
  reconstruction_loss = tf.reduce_mean(reconstruction_err)

  prob_l = tfp.distributions.Normal(loc=mean_locations_t, scale=std_locations)

  log_prob_l = tf.reduce_sum(prob_l.log_prob(locations_t), 2)

  neg_advs = (rewards - tf.stop_gradient(tf.reduce_mean(rewards)))

  reinforce_reconstruction_loss = -tf.reduce_mean(neg_advs * log_prob_l)

  return reconstruction_loss, reinforce_reconstruction_loss


def entropy_loss(probs):
  """Computes entropy of categorical distribution with probabilities (probs)."""
  batch_size = probs.shape.as_list()[0]
  prob = tfp.distributions.Categorical(
      probs=tf.reshape(probs, (batch_size, -1)))
  entropy = prob.entropy()
  return tf.reduce_mean(entropy)


def smoothl1_loss(target_tensor, prediction_tensor,
                  loss_collection=None,
                  reduction=tf.compat.v1.losses.Reduction.NONE):
  """Computes smoothed-l1 loss."""
  return tf.reduce_sum(tf.losses.huber_loss(
      labels=target_tensor,
      predictions=prediction_tensor,
      loss_collection=loss_collection,
      reduction=reduction
      ), axis=-1)


def saccader_pretraining_loss(model, images, is_training):
  """Saccader pretraining loss.

  Args:
    model: Callable saccader model object.
    images: (4D Tensor) input images.
    is_training: (Boolen) training or inference mode.

  Returns:
    Pretraining loss for the model location weights.
  """
  _, _, _, endpoints = model(
      images,
      num_times=12,
      is_training=is_training,
      policy="learned",
      stop_gradient_after_representation=True)

  location_scale = endpoints["location_scale"]
  logits2d = endpoints["logits2d"]
  locations_logits2d_t = endpoints["locations_logits2d_t"]
  batch_size, height, width, _ = logits2d.shape.as_list()
  num_times = len(locations_logits2d_t)
  target_locations_t = saccader.engineered_policies(
      images, logits2d,
      utils.position_channels(logits2d) * location_scale,
      model.glimpse_shape,
      num_times, policy="ordered_logits")

  one_hot_t = []
  for loc in target_locations_t:
    one_hot_t.append(
        tf.reshape(
            utils.onehot2d(
                logits2d,
                tf.stop_gradient(loc) / location_scale),
            (batch_size, height*width))
        )

  locations_logits_t = [tf.reshape(
      locations_logits2d_t[t], (batch_size, height*width))
                        for t in range(num_times)]
  pretrain_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
      onehot_labels=tf.concat(one_hot_t, axis=0),
      logits=tf.concat(locations_logits_t, axis=0),
      loss_collection=None,
      reduction=tf.losses.Reduction.NONE))
  return pretrain_loss
