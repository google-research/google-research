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

"""Create a label map file for Felix and FelixPointer.

Felix label map files are created by converting existing lasertagger
label map files. Laserpointer assumes that if a phrase can be inserted then
all subphrases can also be inserted. This is not the case with lasertagger.

Felix label map file consists of inserting up to N number of MASKS.
"""

import json

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from felix import felix_flags  # pylint: disable=unused-import

FLAGS = flags.FLAGS

# Vocabulary constructions specific flag(s) are listed below. Additionally, we
# use the following flags from nlp/heady/felix/felix_flags.py:
#   - do_lower_case
#   - label_map_file
#   - max_mask
#   - use_pointing

flags.DEFINE_string('output', None,
                    'Path to the resulting new label_map_file.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # always use these entries.
  label_map = {'PAD': 0, 'SWAP': 1, 'KEEP': 2, 'DELETE': 3}
  # Create Insert 1 MASK to insertion N MASKS.
  for i in range(1, FLAGS.max_mask+1):
    label_map[f'KEEP|{i}'] = len(label_map)
    if not FLAGS.use_pointing:
      label_map[f'DELETE|{i}'] = len(label_map)
  logging.info('Created new label map with %d labels', len(label_map))
  with tf.io.gfile.GFile(FLAGS.output, mode='w') as f:
    json.dump(label_map, f)


if __name__ == '__main__':
  flags.mark_flag_as_required('output')
  app.run(main)
