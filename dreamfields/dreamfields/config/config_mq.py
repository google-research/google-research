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

"""Medium quality configuration."""

from dreamfields.config import config_base


def get_config(iters=10000):  # pylint: disable=invalid-name
  """Generate medium quality config dict.

  Args:
    iters (int): Number of training iterations. Some hyperparameter schedules
      are based on this value, as well as the total training duration.

  Returns:
    config (ml_collections.ConfigDict): Configuration object.
  """

  config = config_base.get_config(iters=iters)
  config.render_width = 168
  config.crop_width = 154

  return config
