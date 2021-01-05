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

"""This is the main file to create the toy dataset and clusters."""
#  lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from ipca import n0
import toy_helper


def main(_):
  """Creates the toy dataset main."""
  # Prepares dataset
  width, height = toy_helper.create_dataset()
  # Loads dataset
  x, y, concept = toy_helper.load_xyconcept()
  x_train = x[:n0, :]
  x_val = x[n0:, :]
  y_train = y[:n0, :]
  y_val = y[n0:, :]
  # Loads model
  _, _, feature_dense_model = toy_helper.load_model(
      x_train, y_train, x_val, y_val, pretrain=False)
  toy_helper.create_feature(x, width, height, feature_dense_model)
  # Runs after create_feature
  toy_helper.create_cluster(concept)


if __name__ == '__main__':
  app.run(main)
