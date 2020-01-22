# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Collection of all defined agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from behavior_regularized_offline_rl.brac import bc_agent
from behavior_regularized_offline_rl.brac import bcq_agent
from behavior_regularized_offline_rl.brac import brac_dual_agent
from behavior_regularized_offline_rl.brac import brac_primal_agent
from behavior_regularized_offline_rl.brac import sac_agent


AGENT_MODULES_DICT = {
    'bc': bc_agent,
    'bcq': bcq_agent,
    'sac': sac_agent,
    'brac_primal': brac_primal_agent,
    'brac_dual': brac_dual_agent,
}
