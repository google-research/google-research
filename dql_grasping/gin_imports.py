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

"""Runs imports necessary to register Gin configurables + plugins.

Import this file into top-level python binaries, before parsing configs.
"""

# pylint: disable=unused-import
from gin.tf import external_configurables
from dql_grasping import ddpg_graph
from dql_grasping import episode_to_transitions
from dql_grasping import grasping_env
from dql_grasping import input_data
from dql_grasping import policies
from dql_grasping import q_graph
from dql_grasping import run_env
from dql_grasping import schedules
from dql_grasping import tf_critics
from dql_grasping import train_collect_eval
from dql_grasping import train_ddpg
from dql_grasping import train_q
from dql_grasping import writer


# pylint: enable=unused-import

