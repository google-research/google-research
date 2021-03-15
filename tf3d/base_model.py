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

"""A abstract model with customized training step."""
import tensorflow as tf


class BaseModel(tf.keras.Model):
  """A abstract model with customized training step.

  The subclass should implement the model's forward pass
  in `call` and call `calculates_losses` method during training.
  """

  def __init__(self,
               loss_names_to_functions=None,
               loss_names_to_weights=None,
               train_dir='/tmp/model/train',
               summary_log_freq=100):
    """Initialize the base model.

    Args:
      loss_names_to_functions: A dictionary mapping the loss names to loss
        functions.
      loss_names_to_weights: A dictionary mapping the of loss names to
        loss weights. The losses in this dictionary will be optimized such that
        each loss is weighted with its loss_weight.
      train_dir: A directory path to write tensorboard summary for losses.
      summary_log_freq: An int of the frequency (as batches) to log summary.
    """
    super().__init__()
    if loss_names_to_functions is None:
      loss_names_to_functions = {}
    self.loss_names_to_functions = loss_names_to_functions
    if loss_names_to_weights is None:
      loss_names_to_weights = {}
    self.loss_names_to_weights = loss_names_to_weights

    # loss values per forward run
    self.loss_names_to_losses = {}

    self._batch_update_freq = summary_log_freq
    self._writers = {}
    self._train_dir = train_dir

  @property
  def _train_writer(self):
    """A lazily initialized summary writer for training."""
    if 'train' not in self._writers:
      self._writers['train'] = tf.summary.create_file_writer(
          self._train_dir)
    return self._writers['train']

  def close_writer(self):
    """Close the summary writer.

    This should be called at the end of training.
    """
    for writer in self._writers.values():
      writer.close()

  def calculate_losses(self, inputs, outputs):
    """A function to calculate all losses.

    The function should be called within `call` during training. After the call,
    the losses are available at `self.losses` and `self.loss_names_to_losses`.

    Arguments:
      inputs: A dictionary containing ground-truth tensors
      outputs: A dictionary containing predicted tensors.

    Returns:
      A Tensor representing the (weighted) total loss.
    """

    def get_loss_value(loss_name):
      return self.loss_names_to_functions[loss_name](
          inputs=inputs,
          outputs=outputs) * self.loss_names_to_weights[loss_name]

    loss_values = []
    for loss_name, _ in self.loss_names_to_functions.items():
      loss_value = get_loss_value(loss_name)
      self.loss_names_to_losses[loss_name] = loss_value
      loss_values.append(loss_value)
    total_loss = tf.math.add_n(loss_values)
    self.loss_names_to_losses['total_loss'] = total_loss
    return total_loss

  def train_step(self, data):
    """The logic for one training step.

    This method is overridden to support custom training logic.
    This method is called by `Model.make_train_function`.
    This method should contain the mathematical logic for one step of training.
    This typically includes the forward pass, loss calculation, backpropagation,
    and metric updates.
    Configuration details for *how* this logic is run (e.g. `tf.function` and
    `tf.distribute.Strategy` settings), should be left to
    `Model.make_train_function`, which can also be overridden.

    Arguments:
      data: A dictionary of `Tensor`s.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned. Example:
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """

    # If `sample_weight` is not provided, all samples will be weighted
    # equally.
    x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)

    with tf.GradientTape() as tape:
      self(x, training=True)
      self.write_summary_on_train_batch_end()
      loss = self.loss_names_to_losses['total_loss']
      if tf.executing_eagerly():
        tf.print('total loss:', loss)
      trainable_variables = self.trainable_variables
      gradients = tape.gradient(loss, trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    # TODO(alirezafathi): rewrite following line to add metrics updates
    # self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return {m.name: m.result() for m in self.metrics}

  def write_summary_on_train_batch_end(self):
    """Optionally write scalar summaries of losses and learning rate."""
    with self._train_writer.as_default():
      with tf.summary.record_if(self._train_counter %
                                self._batch_update_freq == 0):
        tf.summary.scalar(
            'learning_rate',
            self.optimizer.learning_rate(self._train_counter),
            step=self._train_counter)
        for loss_name, loss_value in self.loss_names_to_losses.items():
          tf.summary.scalar(loss_name, loss_value, step=self._train_counter)
