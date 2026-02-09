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

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from minisweagent.models import GLOBAL_MODEL_STATS


@dataclass
class DeterministicModelConfig:
    outputs: list[str]
    model_name: str = "deterministic"
    cost_per_call: float = 1.0


class DeterministicModel:
    def __init__(self, **kwargs):
        """
        Initialize with a list of outputs to return in sequence.
        """
        self.config = DeterministicModelConfig(**kwargs)
        self.current_index = -1
        self.cost = 0.0
        self.n_calls = 0

    def query(self, messages, **kwargs):
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if "/sleep" in output:
            print("SLEEPING")
            time.sleep(float(output.split("/sleep")[1]))
            return self.query(messages, **kwargs)
        if "/warning" in output:
            logging.warning(output.split("/warning")[1])
            return self.query(messages, **kwargs)
        self.n_calls += 1
        self.cost += self.config.cost_per_call
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return {"content": output}

    def get_template_vars(self):
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
