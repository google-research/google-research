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

"""Policy Gradient Controller from the original NAS paper."""

# copybara:strip_begin
import pyglove.google as pg
# copybara:strip_end_and_replace
# import pyglove as pg
# copybara:strip_end
from es_enas.controllers import base_controller


class PolicyGradientController(base_controller.BaseController):
  """Policy Gradient Controller."""

  def __init__(self,
               dna_spec,
               batch_size,
               update_batch_size=64,
               **kwargs):
    """Initialization. See base class for more details."""

    super().__init__(dna_spec, batch_size)
    self._controller = pg.generators.policy_gradient.PPO(
        train_batch_size=self._batch_size, update_batch_size=update_batch_size)
    self._controller.setup(self._dna_spec)
    # If you have:
    # training batch size N (PG proposes a batch N of models, stored in cache)
    # update batch size M, (minibatch update batch size)
    # num. of updates P, (how many minibatch updates)
    # the update rule is:
    #
    # for _ in range(P):
    #  mini_batch = select(M, N)
    #  train(model, mini_batch)

