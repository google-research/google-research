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

import os

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.key_per_thread import get_key_per_thread


class AnthropicModel(LitellmModel):
    """For the use of anthropic models, we need to add explicit cache control marks
    to the messages or we lose out on the benefits of the cache.
    Because break points are limited per key, we also need to rotate between different keys
    if running with multiple agents in parallel threads.
    """

    def query(self, messages, **kwargs):
        api_key = None
        if rotating_keys := os.getenv("ANTHROPIC_API_KEYS"):
            api_key = get_key_per_thread(rotating_keys.split("::"))
        return super().query(set_cache_control(messages), api_key=api_key, **kwargs)
