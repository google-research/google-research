# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

from minisweagent import Model
from minisweagent.models import get_model


@dataclass
class RouletteModelConfig:
    model_kwargs: list[dict]
    """The models to choose from"""
    model_name: str = "roulette"


class RouletteModel:
    def __init__(self, *, config_class = RouletteModelConfig, **kwargs):
        """This "meta"-model randomly selects one of the models at every call"""
        self.config = config_class(**kwargs)
        self.models = [get_model(config=config) for config in self.config.model_kwargs]

    @property
    def cost(self):
        return sum(model.cost for model in self.models)

    @property
    def n_calls(self):
        return sum(model.n_calls for model in self.models)

    def get_template_vars(self):
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def select_model(self):
        return random.choice(self.models)

    def query(self, *args, **kwargs):
        model = self.select_model()
        response = model.query(*args, **kwargs)
        response["model_name"] = model.config.model_name
        return response


@dataclass
class InterleavingModelConfig:
    model_kwargs: list[dict]
    sequence: list[int] | None = None
    """If set to 0, 0, 1, we will return the first model 2 times, then the second model 1 time,
    then the first model again, etc."""
    model_name: str = "interleaving"


class InterleavingModel(RouletteModel):
    def __init__(self, *, config_class = InterleavingModelConfig, **kwargs):
        """This "meta"-model alternates between the models in the sequence for every call"""
        super().__init__(config_class=config_class, **kwargs)

    def select_model(self):
        if self.config.sequence is None:
            i_model = self.n_calls % len(self.models)
        else:
            i_model = self.config.sequence[self.n_calls % len(self.config.sequence)]
        return self.models[i_model]
