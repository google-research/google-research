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

# Lint as: python3
"""Generate a json file containing a dictionary mapping names to configs.

This manages sampling configs for both optimizers and tasks.

The tasks are not directly part of the task dataset of tasks as not all configs
will be feasible and/or will train.
"""
import collections
from absl import app
from absl import flags

from task_set import registry
from task_set.optimizers import all_optimizers  # pylint: disable=unused-import
from task_set.tasks import all_tasks  # pylint: disable=unused-import
from task_set.tasks import utils
import tensorflow.compat.v1 as tf

flags.DEFINE_string("task_sampler", None, "Module to use for sampling tasks")
flags.DEFINE_string("optimizer_sampler", None,
                    "Module to use for sampling optimizers")

flags.DEFINE_string("output_file", None, "Output location for sampling.")
flags.mark_flag_as_required("output_file")

flags.DEFINE_integer("num_samples", 100, "Number of samples so select.")
FLAGS = flags.FLAGS


def main(_):
  if FLAGS.task_sampler and FLAGS.optimizer_sampler:
    raise ValueError("Only specify one sampler!")
  if not FLAGS.task_sampler and not FLAGS.optimizer_sampler:
    raise ValueError("Must specify either task_sampler or optimizer_sampler!")

  if FLAGS.task_sampler:
    sampler = registry.task_registry.get_sampler(FLAGS.task_sampler)
    sampler_name = FLAGS.task_sampler
  else:
    sampler = registry.optimizers_registry.get_sampler(FLAGS.optimizer_sampler)
    sampler_name = FLAGS.optimizer_sampler

  samples = collections.OrderedDict()
  for i in range(FLAGS.num_samples):
    cfg = sampler(i)
    task_name = "%s_seed%d" % (sampler_name, i)
    samples[task_name] = cfg, sampler_name

  with tf.gfile.GFile(FLAGS.output_file, "w") as f:
    f.write(utils.pretty_json_dumps(samples).encode("utf-8"))


if __name__ == "__main__":
  app.run(main)
