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

"""Registry of metrics to use."""

import collections
import functools

from ksme.random_mdps import bisimulation
from ksme.random_mdps import mico


MetricData = collections.namedtuple('metric_data', ['constructor', 'label'])


METRICS = {
    'bisimulation': MetricData(bisimulation.Bisimulation, r'$d^{\sim}_{\pi}$'),
    'mico': MetricData(mico.MICo, r'$U^{\pi}$'),
    'reduced_mico': MetricData(functools.partial(mico.MICo, reduced=True),
                               r'$d_{ksme}$'),
}
