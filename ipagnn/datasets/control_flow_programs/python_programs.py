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

"""Functions to transform statements into Python ASTs or CFGs."""

from python_graphs import control_flow
from python_graphs import program_utils


def to_ast(python_source):
  return program_utils.program_to_ast(python_source)


def to_cfg(python_source):
  return control_flow.get_control_flow_graph(python_source)
