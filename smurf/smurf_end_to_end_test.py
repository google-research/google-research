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

"""Tests that training and evaluation work as expected."""

# pylint:skip-file
import contextlib
import io

from absl import flags

from absl.testing import absltest
from smurf import smurf_flags
from smurf import smurf_trainer

FLAGS = flags.FLAGS


class SmurfEndToEndTest(absltest.TestCase):

  def test_training_on_spoof(self):
    FLAGS.eval_on = ''
    FLAGS.train_on = 'spoof:unused'
    FLAGS.plot_dir = '/tmp/spoof_train'
    FLAGS.check_data = True
    FLAGS.num_train_steps = 1
    FLAGS.epoch_length = 1
    FLAGS.evaluate_during_train = False
    FLAGS.height = 296
    FLAGS.width = 296

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
      smurf_trainer.train_eval()

    # Check that the relevant metrics are printed to stdout.
    stdout_message = f.getvalue()
    self.assertIn('total-loss: ', stdout_message)
    self.assertIn('data-time: ', stdout_message)
    self.assertIn('learning-rate: ', stdout_message)
    self.assertIn('train-time: ', stdout_message)

  def test_evaluating_on_spoof(self):
    FLAGS.eval_on = 'spoof:unused'
    FLAGS.check_data = False
    FLAGS.train_on = ''
    FLAGS.plot_dir = '/tmp/spoof_eval'
    FLAGS.height = 296
    FLAGS.width = 296

    FLAGS.num_train_steps = 1
    FLAGS.evaluate_during_train = True
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
      smurf_trainer.train_eval()

    # Check that the relevant metrics are printed to stdout.
    stdout_message = f.getvalue()
    self.assertIn('spoof-EPE: ', stdout_message)
    self.assertIn('spoof-occl-f-max: ', stdout_message)
    self.assertIn('spoof-ER: ', stdout_message)
    self.assertIn('spoof-best-occl-thresh: ', stdout_message)
    self.assertIn('spoof-eval-time(s): ', stdout_message)
    self.assertIn('spoof-inf-time(ms): ', stdout_message)


if __name__ == '__main__':
  absltest.main()
