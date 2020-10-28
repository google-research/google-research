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

# Lint as: python3
"""Simple training runner."""

from absl import app
from absl import flags
import gin

from gfsa.training import simple_runner


flags.DEFINE_string("train_log_dir", None, "Path to log directory.")
flags.DEFINE_string("train_artifacts_dir", None,
                    "Path to save params and other artifacts.")

flags.DEFINE_multi_string("gin_files", [], "Gin config files to use.")
flags.DEFINE_multi_string("gin_include_dirs", [],
                          "Directories to search when resolving gin includes.")

flags.DEFINE_multi_string(
    "gin_bindings", [],
    "Gin bindings to override the values set in the config files.")

flags.DEFINE_enum("task", None, {"maze", "edge_supervision", "var_misuse"},
                  "Task to run.")

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # pylint:disable=g-import-not-at-top
  if FLAGS.task == "maze":
    from gfsa.training import train_maze_lib
    train_fn = train_maze_lib.train
  elif FLAGS.task == "edge_supervision":
    from gfsa.training import train_edge_supervision_lib
    train_fn = train_edge_supervision_lib.train
  elif FLAGS.task == "var_misuse":
    from gfsa.training import train_var_misuse_lib
    train_fn = train_var_misuse_lib.train
  else:
    raise ValueError(f"Unrecognized task {FLAGS.task}")
  # pylint:enable=g-import-not-at-top

  print("Setting up Gin configuration")

  for include_dir in FLAGS.gin_include_dirs:
    gin.add_config_file_search_path(include_dir)

  gin.bind_parameter("simple_runner.training_loop.artifacts_dir",
                     FLAGS.train_artifacts_dir)
  gin.bind_parameter("simple_runner.training_loop.log_dir", FLAGS.train_log_dir)

  gin.parse_config_files_and_bindings(
      FLAGS.gin_files,
      FLAGS.gin_bindings,
      finalize_config=False,
      skip_unknown=False)

  gin.finalize()

  train_fn(runner=simple_runner)


if __name__ == "__main__":
  app.run(main)
