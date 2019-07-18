# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Base class for defining training algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from absl import flags

import tensorflow.compat.v2 as tf

from tcc.config import CONFIG
from tcc.models import get_model
from tcc.utils import get_cnn_feats
from tcc.utils import set_learning_phase

FLAGS = flags.FLAGS


class Algorithm(tf.keras.Model):
  """Base class for defining algorithms."""
  _metaclass_ = abc.ABCMeta

  def __init__(self, model=None):
    super(Algorithm, self).__init__()
    if model:
      self.model = model
    else:
      self.model = get_model()

  @set_learning_phase
  @abc.abstractmethod
  def call(self, data, steps, seq_lens, training):
    """One pass through the model.

    Args:
      data: dict, batches of tensors from many videos. Available keys: 'audio',
      'frames', 'labels'.
      steps: Tensor, batch of indices of chosen frames in videos.
      seq_lens: Tensor, batch of sequence length of the full videos.
      training: Boolean, if True model is run in training mode.

    Returns:
      embeddings: Tensor, Float tensor containing embeddings

    Raises:
      ValueError: In case invalid configs are passed.
    """
    cnn = self.model['cnn']
    emb = self.model['emb']

    if training:
      num_steps = CONFIG.TRAIN.NUM_FRAMES
    else:
      num_steps = CONFIG.EVAL.NUM_FRAMES

    cnn_feats = get_cnn_feats(cnn, data, training)

    embs = emb(cnn_feats, num_steps)
    channels = embs.shape[-1]
    embs = tf.reshape(embs, [-1, num_steps, channels])

    return embs

  @abc.abstractmethod
  def compute_loss(self, embs, steps, seq_lens, global_step, training,
                   frame_labels=None, seq_labels=None):
    pass

  def get_base_and_embedding_variables(self):
    """Gets list of trainable vars from model's base and embedding networks.

    Returns:
      variables: List, list of variables we want to train.
    """

    if CONFIG.MODEL.TRAIN_BASE == 'train_all':
      variables = self.model['cnn'].variables
    elif CONFIG.MODEL.TRAIN_BASE == 'only_bn':
      # TODO(debidatta): Better way to extract batch norm variables.
      variables = [x for x in self.model['cnn'].variables
                   if 'batch_norm' in x.name or 'bn' in x.name]
    elif CONFIG.MODEL.TRAIN_BASE == 'frozen':
      variables = []
    else:
      raise ValueError('train_base values supported right now: train_all, '
                       'only_bn or frozen.')
    if CONFIG.MODEL.TRAIN_EMBEDDING:
      variables += self.model['emb'].variables
    return variables

  @abc.abstractmethod
  def get_algo_variables(self):
    return []

  @property
  def variables(self):
    """Returns list of variables to train.

    Returns:
      variables: list, Contains variables that will be trained.
    """
    variables = [x for x in self.get_base_and_embedding_variables()
                 if 'moving' not in x.name]
    variables += [x for x in  self.get_algo_variables()
                  if 'moving' not in x.name]
    return variables

  def compute_gradients(self, loss, tape=None):
    """This is to be used in Eager mode when a GradientTape is available."""
    if tf.executing_eagerly():
      assert tape is not None
      gradients = tape.gradient(loss, self.variables)
    else:
      gradients = tf.gradients(loss, self.variables)
    return gradients

  def apply_gradients(self, optimizer, grads):
    """Functional style apply_grads for `tfe.defun`."""
    optimizer.apply_gradients(zip(grads, self.variables))

  def train_one_iter(self, data, steps, seq_lens, global_step, optimizer):
    with tf.GradientTape() as tape:
      embs = self.call(data, steps, seq_lens, training=True)
      loss = self.compute_loss(embs, steps, seq_lens, global_step,
                               training=True, frame_labels=data['frame_labels'],
                               seq_labels=data['seq_labels'])
      # Add regularization losses.
      reg_loss = tf.reduce_mean(tf.stack(self.losses))
      tf.summary.scalar('reg_loss', reg_loss, step=global_step)
      loss += reg_loss

      # Be careful not to use object based losses in tf.keras.losses
      # (CategoricalCrossentropy) or tf.losses (softmax_cross_entropy). The
      # above losses scale by number of GPUs on their own which can lead to
      # inconsistent scaling. Hence, always use functional version losses
      # defined in tf.keras.losses (categorical_crossentropy.
      # Divide by number of replicas.
      strategy = tf.distribute.get_strategy()
      num_replicas = strategy.num_replicas_in_sync
      loss *= (1. / num_replicas)

    gradients = self.compute_gradients(loss, tape)
    self.apply_gradients(optimizer, gradients)

    if FLAGS.debug:
      for v, g in zip(self.variables, gradients):
        norm = tf.reduce_sum(g*g)
        tf.summary.scalar('grad_norm_%s' % v.name, norm,
                          step=global_step)
      grad_norm = tf.reduce_mean(tf.stack([tf.reduce_sum(grad * grad)
                                           for grad in gradients]))
      tf.summary.scalar('grad_norm', grad_norm, step=global_step)
      for k in self.model:
        for var_ in self.model[k].variables:
          tf.summary.histogram(var_.name, var_)

    return loss
