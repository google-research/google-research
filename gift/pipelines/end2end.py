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
"""End2end trainer."""

from flax.deprecated import nn
from gift.pipelines import trainer


class End2end(trainer.Trainer):
  """End2end backpropagation."""

  def training_loss_fn(self, flax_module, train_state, batch, dropout_rng):
    """Runs forward pass and computes loss.

    Args:
      flax_module: A flax module.
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer.
      batch: Batches from different environments.
      dropout_rng: FLAX PRNG key.

    Returns:
      loss, new_module_state and computed logits for each batch.
    """
    with nn.stateful(train_state.model_state) as new_model_state:
      with nn.stochastic(dropout_rng):
        logits = flax_module(batch['inputs'], train=True)

    loss = self.task.loss_function(logits, batch, flax_module.params)
    return loss, (new_model_state, logits)
