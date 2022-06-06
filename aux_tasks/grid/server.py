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

r"""Successor Representation replay buffer, part of distributed learning setup.

Example command:

python -m aux_tasks.grid.server \
  --port=1234 \
  --config=aux_tasks/grid/config.py:implicit


"""

from collections.abc import Sequence

from absl import app
from absl import flags
from ml_collections import config_flags
import reverb

flags.DEFINE_integer('port', None, 'Port to start the server on.')
_CONFIG = config_flags.DEFINE_config_file('config', lock_config=True)

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  eval_table_size = _CONFIG.value.num_eval_points

  # TODO(joshgreaves): Choose an appropriate rate_limiter, max_size.
  server = reverb.Server(
      tables=[
          reverb.Table(
              name='successor_table',
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              max_size=1_000_000,
              rate_limiter=reverb.rate_limiters.MinSize(20_000)),
          reverb.Table(
              name='eval_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              max_size=eval_table_size,
              rate_limiter=reverb.rate_limiters.MinSize(eval_table_size)),
      ],
      port=FLAGS.port)
  server.wait()


if __name__ == '__main__':
  app.run(main)
