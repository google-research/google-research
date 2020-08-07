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

"""Supporting utilities for computing associations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from absl import flags

flags.DEFINE_string(
    "association_dir", "data/train",
    "Directory where the association files, such as raw proportions and "
    "implicationals reside.")

flags.DEFINE_string(
    "genus_filename", "raw_proportions_by_genus.tsv",
    "Output for genus-feature clade affiliations.")

flags.DEFINE_string(
    "family_filename", "raw_proportions_by_family.tsv",
    "Output for family-feature clade affiliations.")

flags.DEFINE_string(
    "neighborhood_filename", "raw_proportions_by_neighborhood.tsv",
    "Output for neighborhood-feature affiliations.")

flags.DEFINE_string(
    "implicational_filename", "implicational_universals.tsv",
    "Output for putative implicational universals.")

FLAGS = flags.FLAGS
