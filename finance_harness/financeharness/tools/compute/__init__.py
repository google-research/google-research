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

"""Compute tools — pure-math, vendor-agnostic valuation.

No network: compute tools consume numbers (often chained from data tools via
`prev:<id>.<path>`) and return derived measures. Stateless deferred SPECs.
"""

from financeharness.tools.compute.risk import RISK_SPECS
from financeharness.tools.compute.valuation import VALUATION_SPECS

COMPUTE_SPECS = [*VALUATION_SPECS, *RISK_SPECS]

__all__ = ["COMPUTE_SPECS", "RISK_SPECS", "VALUATION_SPECS"]
