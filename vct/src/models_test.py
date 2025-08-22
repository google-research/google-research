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

"""Tests for models."""

import functools

import tensorflow as tf

from vct.src import models
from vct.src import video_tensors


ModelLite = functools.partial(models.Model, lightweight=True)


class ModelsTest(tf.test.TestCase):

  def _restore_evaluate(self, ckpt_p):
    """Restore and evaluate a model for the given stage."""
    model = ModelLite()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(ckpt_p).assert_existing_objects_matched()
    video = video_tensors.EvalVideo.make_random(num_frames=3)
    count = 0
    for metrics in model.evaluate(video):
      count += 1
      self.assertIn("bpp", metrics.scalars)
    self.assertEqual(count, 3)

  def test_train_eval(self):
    ckpt_dir = self.create_tempdir().full_path
    model = ModelLite()
    train_step = tf.function(model.train_step)
    train_step(video_tensors.TrainingVideo.make_random(num_frames=3))
    ckpt_p = model.write_ckpt(ckpt_dir, step=1)
    self._restore_evaluate(ckpt_p)


if __name__ == "__main__":
  tf.test.main()
