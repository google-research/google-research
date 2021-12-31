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

"""The version of the control_flow_programs dataset."""

VERSION = "0.0.55"
MINIMUM_SUPPORTED_VERSION = "0.0.40"


def as_tuple(version_str):
  return tuple(int(part) for part in version_str.split("."))


def at_least(version_str):
  return as_tuple(VERSION) >= as_tuple(version_str)


def supports_edge_types():
  return at_least("0.0.47")
