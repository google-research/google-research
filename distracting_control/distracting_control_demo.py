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

"""A simple demo that produces an image from the environment."""
import os
from absl import app
from absl import flags

import PIL

from distracting_control import suite

FLAGS = flags.FLAGS

# NOTE: This has been populated with the demo images path, but you should
# download and extract the DAVIS dataset and point this path to the location
# of DAVIS.
flags.DEFINE_string(
    'davis_path',
    '',
    'Path to DAVIS images, used for background distractions.')
flags.DEFINE_string('output_dir', '/tmp/distracting_control_demo',
                    'Directory where the results are being saved.')


def main(unused_argv):

  if not FLAGS.davis_path:
    raise ValueError(
        'You must download and extract the DAVIS dataset and pass a path '
        'a path to the videos, e.g.: /tmp/DAVIS/JPEGImages/480p')

  for i, difficulty in enumerate(['easy', 'medium', 'hard']):
    for j, (domain, task) in enumerate([
        ('ball_in_cup', 'catch'),
        ('cartpole', 'swingup'),
        ('cheetah', 'run'),
        ('finger', 'spin'),
        ('reacher', 'easy'),
        ('walker', 'walk')]):

      env = suite.load(
          domain, task, difficulty, background_dataset_path=FLAGS.davis_path)

      # Get the first frame.
      time_step = env.reset()
      frame = time_step.observation['pixels'][:, :, 0:3]

      # Save the first frame.
      try:
        os.mkdir(FLAGS.output_dir)
      except OSError:
        pass
      filepath = os.path.join(FLAGS.output_dir, f'{i:02d}-{j:02d}.jpg')
      image = PIL.Image.fromarray(frame)
      image.save(filepath)
  print(f'Saved results to {FLAGS.output_dir}')


if __name__ == '__main__':
  app.run(main)
