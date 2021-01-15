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

"""Regularized Evolution controller from PyGlove. Similar to NEAT."""
import numpy as np
import pyglove as pg
from es_enas.controllers import base_controller


class RegularizedEvolutionController(base_controller.BaseController):
  """Regularized Evolution Controller."""

  def __init__(self, dna_spec, batch_size,
               **kwargs):
    """Initialization. See base class for more details."""

    super().__init__(dna_spec, batch_size)
    population_size = self._batch_size
    tournament_size = int(np.sqrt(population_size))

    self._controller = pg.generators.RegularizedEvolution(
        population_size=population_size,
        tournament_size=tournament_size,
        mutator=pg.generators.evolution_mutators.Uniform())
    self._controller.setup(self._dna_spec)
