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
"""Example using opt_list with pytorch."""
from absl import app

import numpy as np
from opt_list import torch_opt_list
import torch


def main(_):
  # Construct the model
  model = torch.nn.Sequential(
      torch.nn.Linear(2, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
      torch.nn.ReLU(), torch.nn.Linear(256, 2))
  loss_fn = torch.nn.MSELoss(reduction="mean")

  # Define how long to train for
  training_iters = 200

  # Create the optimizer corresponding to the 0th hyperparameter configuration
  # with the specified amount of training steps. The result follows the
  # same API as the other torch optimizers (e.g. Adam).

  # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  optimizer = torch_opt_list.optimizer_for_idx(model.parameters(), 0,
                                               training_iters)

  for _ in range(training_iters):
    # Make a batch of randomly generated data.
    inp = np.random.normal([512, 2]).astype(np.float32) / 4.
    target = np.tanh(1 / (1e-6 + inp))

    # Forward pass + loss computation
    y_pred = model(torch.tensor(inp))
    loss = loss_fn(y_pred, torch.tensor(target))

    # perform a training step.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.detach().numpy())


if __name__ == "__main__":
  app.run(main)
