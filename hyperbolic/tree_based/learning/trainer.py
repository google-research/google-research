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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module to train CF embedding models."""
import tensorflow.compat.v2 as tf

from hyperbolic.tree_based.learning import tree_losses as losses


class CFTrainer(object):
  """CF embedding trainer object."""

  def __init__(self, sizes, args):
    """Initialize CF trainer.

    Args:
      sizes: Tuple of size 2 containing (n_users, n_items).
      args: Namespace with config arguments (see config.py for detailed overview
        of arguments supported).
    """
    if args.optimizer == 'Adagrad':
      self.optimizer = tf.keras.optimizers.Adagrad(
          learning_rate=args.lr, initial_accumulator_value=0.0, epsilon=1e-10)
    elif args.optimizer == 'Adam':
      self.optimizer = tf.keras.optimizers.Adam(
          learning_rate=args.lr,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-08,
          amsgrad=False)
    else:
      self.optimizer = getattr(tf.keras.optimizers, args.optimizer)(
          learning_rate=args.lr)
    self.loss_fn = getattr(losses, args.loss_fn)(sizes, args)
    self.lr_decay = args.lr_decay
    self.min_lr = args.min_lr

  def reduce_lr(self,):
    """Reduces learning rate."""
    old_lr = float(tf.keras.backend.get_value(self.optimizer.lr))
    if old_lr > self.min_lr:
      new_lr = old_lr * self.lr_decay
      new_lr = max(new_lr, self.min_lr)
      tf.keras.backend.set_value(self.optimizer.lr, new_lr)

  def valid_step(self, model, examples):
    """Computes validation loss.

    Args:
      model: tf.keras.Model CF embedding model.
      examples: tf.data.Dataset containing KG validation triples.

    Returns:
      Average validation loss.
    """
    total_loss = tf.keras.backend.constant(0.0)
    counter = tf.keras.backend.constant(0.0)
    for input_batch in examples:
      counter += 1.0
      total_loss += self.loss_fn.calculate_loss(model, input_batch)
    return total_loss / counter

  @tf.function
  def train_step(self, model, examples):
    """Compute training loss and back-propagate gradients.

    Args:
      model: tf.keras.Model KG embedding model.
      examples: tf.data.Dataset containing CF training pairs.

    Returns:
      Average training loss.
    """
    total_loss = tf.keras.backend.constant(0.0)
    counter = tf.keras.backend.constant(0.0)
    for input_batch in examples:
      counter += 1.0
      with tf.GradientTape() as tape:
        loss = self.loss_fn.calculate_loss(model, input_batch)
      gradients = zip(
          tape.gradient(loss, model.trainable_variables),
          model.trainable_variables)
      self.optimizer.apply_gradients(gradients)
      total_loss += loss
    return total_loss / counter
