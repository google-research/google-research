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
"""For creating files from {target,prediction}.txt that can be processed
by pyrouge to compare with scores in scoring_test.py.

  create_pyrouge_files -- --testdata_dir=`pwd`/testdata

  # testConfidenceIntervalsAgainstRouge155WithStemming result
  pyrouge_evaluate_plain_text_files \
      -s /tmp/lkj -sfp "prediction.(.*).txt" \
      -m /tmp/lkj -mfp target.#ID#.txt

  pyrouge_evaluate_plain_text_files \
      -s /tmp/lkj -sfp "prediction_multi.(.*).txt" \
      -m /tmp/lkj -mfp target_multi.#ID#.txt
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('testdata_dir', '', 'testdata path')
flags.DEFINE_string('output',  '/tmp/lkj', 'testdata path')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # One line per target
  with open(os.path.join(FLAGS.testdata_dir, 'target_large.txt')) as f:
    targets = f.readlines()
  with open(os.path.join(FLAGS.testdata_dir, 'prediction_large.txt')) as f:
    predictions = f.readlines()

  def write_files(prefix, items):
    for i, t in enumerate(items):
      out = '%s.%d.txt' % (prefix, i)
      with open(os.path.join(FLAGS.output, out), 'w') as f:
        f.write(t)
  write_files('target', targets)
  write_files('prediction', predictions)

  # Delete this block
  def write_files2(prefix, items):
    index = 0
    f = None
    for i, t in enumerate(items):
      # Write 4 lines per file
      if i % 4 == 0:
        if f:
          f.close()
        f = open(
            os.path.join(FLAGS.output, '%s.%d.txt' % (prefix, index)),
            'w')
        index += 1
      f.write(t)
    f.close()
  write_files2('target_multi', targets)
  write_files2('prediction_multi', predictions)


if __name__ == '__main__':
  app.run(main)
