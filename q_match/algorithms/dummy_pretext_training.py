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

"""Dummy pretext training algo that can be used as a template.
"""

from absl import logging

from flax.core import freeze
import jax

from q_match.algorithms.training_algo import TrainingAlgo


class DummyPretextTraining(TrainingAlgo):
  """Prextext Training Algorithm.

  Attributes:
    logdir: location of the log directory.
    dataset: tf dataset to train.
    batch_size: batch size for training.
    model: the pretext model to train.
    eval_model: eval model
    learning_rate: the learning rate for training.
    epochs: number of epochs to train for
    params: Optional params to start training from.  If None, random params
      are initialized.
    state: Optional state to start training from.
    writer: Writer for writing to tensorboard.
    weight_decay: weight decay on pretext params.
  """

  def _loss(self, params, state, features):
    """dummy loss which shrinks predictions."""
    variables = freeze({'params': params, **state})
    output, updated_state = self.model.apply(variables, features,
                                             mutable=['batch_stats'],
                                             rngs=self.rngs)
    preds = output['pretext']['encoded']
    return jax.numpy.mean(preds), updated_state

  def run(self,):
    """Runs a pretext training algo."""
    params = self.params
    state = self.state
    dataset = self.dataset
    model = self.model

    if params is None and state is None:
      example_data = jax.numpy.array(dataset.get_example_features())
      model_key = jax.random.PRNGKey(0)
      variables = model.init(model_key, example_data)
      state, params = variables.pop('params')

      example_output, _ = model.apply(variables,
                                      example_data,
                                      mutable=['batch_stats'],
                                      rngs=self.rngs)
      log_msg = str(example_output)
      logging.debug(log_msg)

    optimizer_state = self.optimizer.init(params=params)

    grad_fn = self.get_grad_fn()

    steps = 0
    for epoch in range(self.epochs):
      logging.info('Pretext Epoch: %d', epoch)
      for example in dataset.get_pretext_ds():
        features = jax.numpy.array(example['features'])

        if steps % 100 == 0:
          train_loss = self.loss(params, state, features)[0]
          log_train_loss_msg = f'dummy pretext training loss {train_loss}'
          logging.info(log_train_loss_msg)

          if self.writer is not None:
            self.writer.write_scalars(steps, {'pretext_train_loss': train_loss})

        gradients, state = grad_fn(params, state, features)
        params, optimizer_state = self.update_model(params,
                                                    gradients,
                                                    optimizer_state)
        self.update_rngs()
        steps += 1

    return params, state
