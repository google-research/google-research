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

"""Purely supervised training.

Model must have outputs of form:
  {'main': {'logits': ...}}

for classification.
"""

import copy
import functools

from absl import logging
from flax.core import freeze
import jax

from q_match.algorithms.training_algo import TrainingAlgo
from q_match.tasks.tasks import eval_test_dataset


def cross_entropy(logits, target_class):
  logprobs = jax.nn.log_softmax(logits)
  nll = jax.numpy.take_along_axis(logprobs,
                                  jax.numpy.expand_dims(target_class, axis=-1),
                                  axis=-1)
  ce = -jax.numpy.mean(nll)
  return ce


class SupervisedTraining(TrainingAlgo):
  """Supervised Training Algorithm.

  Attributes:
    logdir: location of the log directory.
    dataset: tf dataset to train.
    batch_size: batch size for training.
    model: the flax model to train.
    eval_model: the eval/inference version of the flax model.
    learning_rate: the learning rate for training.
    epochs: number of epochs to train for
    params: Optional params to start training from.  If None, random params
      are initialized.
    state: Optional state to start training from.
    writer: Writer for writing to tensorboard.
    optimizer: Optimizer for training using gradients.
    optimizer_state: State of the optimizer.
    patience: Number of epochs to wait for improvement on validation dataset.
    weight_decay: Weight decay to apply to parameters.
    linear_head: Whether to use just the linear head (which has a s.g.) applied
      to its inputs) or the fully finetune the network.
    linear_over_features: Wether to compute the linear head over the inputs.
  """

  def __init__(self,
               logdir,
               dataset,
               batch_size,
               model,
               eval_model,
               learning_rate,
               epochs,
               params=None,
               state=None,
               writer=None,
               patience=16,
               weight_decay=0.,
               linear_head=False,
               linear_over_features=False):
    weight_decay_mask = None
    params = copy.deepcopy(params)
    state = copy.deepcopy(state)
    # Only decay linear head params
    if linear_head:
      weight_decay_mask = self.generate_parameter_ancestors(
          params, 'linear_head')
    super(SupervisedTraining, self).__init__(
        logdir,
        dataset,
        batch_size,
        model,
        eval_model,
        learning_rate,
        epochs,
        params,
        state,
        writer,
        weight_decay,
        weight_decay_mask=weight_decay_mask)
    self.patience = patience
    self.patience_counter = 0
    self.early_stop_params = self.params
    self.early_stop_state = self.state
    self.best_early_stop_accuracy = float('-inf')
    self.weight_decay = weight_decay
    self.linear_head = linear_head
    self.linear_over_features = linear_over_features
    if self.linear_head:
      self.write_prefix = 'Linear'
    else:
      self.write_prefix = 'FineTune'

  def _loss(self, params, state, features, targets):
    variables = freeze({'params': params, **state})
    output, updated_state = self.model.apply(
        variables,
        features,
        linear_over_features=self.linear_over_features,
        mutable=['batch_stats'],
        rngs=self.rngs)
    if self.linear_head:
      logits = output['main']['linear_head']['logits']
    else:
      logits = output['main']['finetune_head']['logits']
    return cross_entropy(logits, targets), updated_state

  def run(self,):
    """Trains model."""
    dataset = self.dataset
    model = self.model
    eval_model = self.eval_model

    example_data = jax.numpy.array(dataset.get_example_features())

    if self.params is None and self.state is None:
      model_key = jax.random.PRNGKey(0)
      variables = model.init(model_key, example_data)
      state, params = variables.pop('params')

      example_output, _ = model.apply(variables,
                                      example_data,
                                      mutable=['batch_stats'],
                                      rngs=self.rngs)
      log_msg = f'example output: {example_output}'
      logging.debug(log_msg)
    else:
      params = self.params
      state = self.state

    optimizer_state = self.optimizer.init(params=params)

    @functools.partial(jax.jit, static_argnums=(4,))
    def compute_accuracy(params, state, features, targets,
                         linear_over_features):
      variables = freeze({'params': params, **state})
      output = eval_model.apply(
          variables, features, linear_over_features=linear_over_features)
      if self.linear_head:
        preds = output['main']['linear_head']['logits']
      else:
        preds = output['main']['finetune_head']['logits']

      return jax.numpy.mean(jax.numpy.argmax(preds, axis=-1) == targets)

    grad_fn = self.get_grad_fn()

    steps = 0
    for epoch in range(self.epochs):
      logging.info('Supervised Epoch %d', epoch)
      for example in dataset.get_train_ds():
        features = jax.numpy.array(example['features'])
        targets = jax.numpy.array(example['target'])

        if steps % 100 == 0:
          train_loss = self.loss(params, state, features, targets)[0]
          train_accuracy = compute_accuracy(params, state, features, targets,
                                            self.linear_over_features)
          log_train_loss_msg = f'training loss {train_loss}'
          log_accuracy_msg = f'accuracy: {train_accuracy}'
          logging.info(log_train_loss_msg)
          logging.info(log_accuracy_msg)

          if self.writer is not None:
            self.writer.write_scalars(steps,
                                      {'train_loss': train_loss,
                                       'train_accuracy': train_accuracy})

        gradients, state = grad_fn(params, state, features, targets)
        params, optimizer_state = self.update_model(params,
                                                    gradients,
                                                    optimizer_state)
        self.update_rngs()
        steps += 1

      # Validation check
      validation_results = eval_test_dataset(
          eval_model=self.eval_model,
          finetune_params=params,
          finetune_state=state,
          linear_params=params,
          linear_state=state,
          test_ds=dataset.get_validation_ds(),
          linear_over_features=self.linear_over_features)
      if self.linear_head:
        validation_accuracy = validation_results['linear_accuracy']
      else:
        validation_accuracy = validation_results['finetune_accuracy']
      self.writer.write_scalars(
          epoch,
          {self.write_prefix + '_validation_accuracy': validation_accuracy})
      logging.info('Validation accuracy: %f', validation_accuracy)
      if validation_accuracy > self.best_early_stop_accuracy:
        self.best_early_stop_accuracy = validation_accuracy
        self.early_stop_params = params
        self.early_stop_state = state
        self.patience_counter = 0
      else:
        self.patience_counter += 1

      if self.patience_counter > self.patience:
        break

    return self.early_stop_params, self.early_stop_state
