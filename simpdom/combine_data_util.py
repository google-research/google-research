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

"""The util functions to combine data from different seed websites."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from absl import logging
import tensorflow.compat.v1 as tf


def generate_website_lists(seed_num,
                           cur_website_name_list,
                           goldmine_folder,
                           concat_websites_func=None,
                           domtree_data_path=None,
                           experiment_vertical=None):
  """Generates the source/target website lists, (and combine the files).

  Args:
    seed_num: the number of the seed websites for training.
    cur_website_name_list: a list of website names for the current vertical.
    goldmine_folder: the goldmine data path.
    concat_websites_func: the function used for combine the files for training.
    domtree_data_path: the domtree data path for reading and saving files.
    experiment_vertical: the current vertical we are processing.

  Returns:
    all_source_websites: the list of source website names.
    all_target_websites: the list of target website names.
  """
  if seed_num == 1:
    all_source_websites = cur_website_name_list
    all_target_websites = ["_".join(cur_website_name_list)
                          ] * len(cur_website_name_list)
  else:
    all_source_websites = []
    all_target_websites = []
    extended_websites = cur_website_name_list + cur_website_name_list[:seed_num]
    for round_index in range(len(cur_website_name_list)):
      source_website_set = set(extended_websites[round_index:round_index +
                                                 seed_num])
      target_website_set = set(cur_website_name_list) - source_website_set
      target_website_list = sorted(list(target_website_set))
      source_website_list = extended_websites[round_index:round_index +
                                              seed_num]
      if concat_websites_func and domtree_data_path and experiment_vertical:
        concat_websites_func(domtree_data_path, experiment_vertical,
                             source_website_list, target_website_list,
                             goldmine_folder)
      all_source_websites.append("-".join(source_website_list))
      all_target_websites.append("_".join(target_website_list))
  return all_source_websites, all_target_websites


def read_and_concat_files(input_data_path,
                          output_path,
                          website_list,
                          vertical,
                          cutoff=None,
                          goldmine=False):
  """Reads the data in website_list and concate them to write a single file.

  Args:
    input_data_path: the domtree data path for reading features.
    output_path: the domtree data path for saving features.
    website_list: the list of websites we are processing.
    vertical: the current vertical we are processing.
    cutoff: the number of webpages to keep for each website, all pages will be
      kept when not set.
    goldmine: if to process goldmine features.
  """

  if goldmine:
    concat_res = []
  else:
    concat_res = dict(features=[])
  if not tf.gfile.Exists(output_path):
    logging.info("Concat multiple websites to a single one....")
    for website in website_list:
      if goldmine:
        read_json_path = os.path.join(
            input_data_path, "{}-{}.feat.json".format(vertical, website))
      else:
        read_json_path = os.path.join(input_data_path,
                                      "{}-{}.json".format(vertical, website))

      with tf.gfile.Open(read_json_path) as json_file:
        current_website_data = json.load(json_file)
        if goldmine:
          concat_res += current_website_data[:cutoff]
        else:
          concat_res["features"] += current_website_data["features"][:cutoff]

    with tf.gfile.Open(output_path, "w") as json_file:
      json.dump(concat_res, json_file)
      logging.info("Done creating the concat: %s.", json_file.name)
  else:
    logging.info("%s already exists", output_path)


def concat_websites(domtree_data_path, vertical, source_website_list,
                    target_website_list, goldmine_folder):
  """Concatenates multiple websites as a single one.

  Args:
    domtree_data_path: the domtree data path for reading and saving files.
    vertical: the current vertical we are processing.
    source_website_list: a list of website names for the source vertical.
    target_website_list: a list of website names for the target vertical.
    goldmine_folder: the goldmine data path.
  """

  output_res_path = os.path.join(
      domtree_data_path, "{}-{}.json".format(vertical,
                                             "-".join(source_website_list)))
  output_res_dev_path = os.path.join(
      domtree_data_path, "{}-{}.dev.json".format(vertical,
                                                 "-".join(source_website_list)))

  read_and_concat_files(
      domtree_data_path,
      output_res_path,
      source_website_list,
      vertical,
      cutoff=None)
  read_and_concat_files(
      domtree_data_path,
      output_res_dev_path,
      target_website_list,
      vertical,
      cutoff=50)

  output_goldmine_path = output_res_path.replace(domtree_data_path,
                                                 goldmine_folder).replace(
                                                     ".json", ".feat.json")
  output_goldmine_dev_path = output_res_dev_path.replace(
      domtree_data_path, goldmine_folder).replace(".json", ".feat.json")

  read_and_concat_files(
      goldmine_folder,
      output_goldmine_path,
      source_website_list,
      vertical,
      cutoff=None,
      goldmine=True)

  read_and_concat_files(
      goldmine_folder,
      output_goldmine_dev_path,
      target_website_list,
      vertical,
      cutoff=50,
      goldmine=True)
