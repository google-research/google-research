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

"""Locations of SMURF datasets."""

# pylint:skip-file

# Anyone using this codebase should re-generate the data themselves
# and populate this dictionary with the filepath to the relevant datasets.
# pylint:disable=duplicate-key
dataset_locations = {
    # Sintel.
    'sintel-train': '',
    'sintel-train-clean': '',
    'sintel-train-final': '',
    'sintel-test': '',
    'sintel-test-clean': '',
    'sintel-test-final': '',
    # Flying chairs
    'chairs-all': '',
    'chairs-train': '',
    'chairs-test': '',
    # Kitti12
    'kitti12-train-pairs': '',
    'kitti12-test-pairs': '',
    # KITTI15 (train and test pairs, multiview extensions for both)
    'kitti15-train-pairs': '',
    'kitti15-test-pairs': '',
    'kitti15-train-multiview': '',
    'kitti15-test-multiview': '',
}
