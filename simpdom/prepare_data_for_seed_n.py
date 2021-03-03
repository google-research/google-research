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

"""The binary to combine data from different seed websites."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app
from absl import flags
from simpdom import combine_data_util
from simpdom import constants

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed_num", 1, "Number of the seed websites for training.")
flags.DEFINE_string(
    "domtree_data_path", "",
    "The path to the folder of all tree data formatted of SWDE dataset.")
flags.DEFINE_string(
    "goldmine_data_path", "",
    "The path to the folder of all the node-level goldmine features.")
flags.DEFINE_string("vertical", "", "The vertical to run.")


def main(_):
  vertical_to_websites_map = constants.VERTICAL_WEBSITES
  verticals = [FLAGS.vertical
              ] if FLAGS.vertical else vertical_to_websites_map.keys()
  for vertical in verticals:
    website_list = vertical_to_websites_map[vertical]
    combine_data_util.generate_website_lists(FLAGS.seed_num, website_list,
                                             FLAGS.goldmine_data_path,
                                             combine_data_util.concat_websites,
                                             FLAGS.domtree_data_path, vertical)


if __name__ == "__main__":
  app.run(main)
