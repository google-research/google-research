# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

r"""Running the IMP pipeline.

Example usage:
python3 -m max.projects.imp.main \
  --config_name=name_of_config_of_interest \
  --config_overrides='{path: "path/to/dir"}'

"""

from typing import Sequence

from absl import app
from absl import flags

from imp.max.execution import main as exec_main
from imp.max.projects.imp.config import experiment

FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused

  exec_main.run(
      config_name=FLAGS.config_name,
      config_overrides=FLAGS.config_overrides,
      vizier_study=FLAGS.vizier_study,
      vizier_tuner_group=FLAGS.vizier_tuner_group,
      tf_data_service_address=FLAGS.tf_data_service_address)

if __name__ == '__main__':
  exec_main.define_flags()
  exec_main.ensure_registered(experiment)
  exec_main.initialize_devices()
  app.run(main)
