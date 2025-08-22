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

from collections.abc import Sequence
import json
import os

from absl import app
from act.config.flags import FLAGS, initialize_flags
from act.config.utils import (
    get_config_from_dict,
    get_config_from_flags,
    get_config_from_json_path,
    get_default_config,
)
from act.data.utils import get_datasets_from_config
from act.models.utils import initialize_env
from act.utils.storage_utils import (
    read_json,
    write_jsonl,
)

initialize_flags(get_default_config())

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.config != "":
    act_config = get_config_from_json_path(FLAGS.config)
  else:
    act_config = get_config_from_flags(FLAGS)

  if act_config.training_config.is_preference:
    raise app.Error("The dataset is already preference!")

  initialize_env()

  train_dataset, validation_dataset = get_datasets_from_config(act_config)

  write_jsonl(
      os.path.join(
          act_config.training_config.output_dir,
          act_config.data_config.train_preference_filename,
      ),
      train_dataset,
  )

  write_jsonl(
      os.path.join(
          act_config.training_config.output_dir,
          act_config.data_config.validation_preference_filename,
      ),
      validation_dataset,
  )


if __name__ == "__main__":
  app.run(main)
