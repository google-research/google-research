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

"""Data tools — vendor-agnostic interfaces, yfinance implementation.

Data tools are stateless (unlike the research trio), so they're module-level
``ToolSpec`` constants registered into a registry. Tier is ``deferred``: they're
catalog-only until ``load_tool`` (or a skill) pulls them, keeping the
orchestrator's context lean.
"""
