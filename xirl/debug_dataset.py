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

"""Debugging script to visualize the dataset video and frame sampling."""

import sys

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
from ml_collections import config_flags
import torchvision
from xirl.common import get_pretraining_dataloaders
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_boolean("debug", False, "Debug mode.")

config_flags.DEFINE_config_file(
    "config",
    "configs/pretrain_default.py",
    "File path to the training hyperparameter configuration.",
)


def main(_):
  num_ctx_frames = FLAGS.config.FRAME_SAMPLER.NUM_CONTEXT_FRAMES
  num_frames = FLAGS.config.FRAME_SAMPLER.NUM_FRAMES_PER_SEQUENCE
  pretrain_loaders = get_pretraining_dataloaders(FLAGS.config, FLAGS.debug)
  try:
    loader = pretrain_loaders["train"]
    logging.info(f"Total videos: {loader.dataset.total_vids}")
    for batch_idx, batch in enumerate(loader):
      logging.info(f"Batch #{batch_idx}")
      frames = batch["frames"]
      b, _, c, h, w = frames.shape
      frames = frames.view(b, num_frames, num_ctx_frames, c, h, w)
      for b in range(frames.shape[0]):
        logging.info(f"\tBatch Item {b}")
        grid_img = torchvision.utils.make_grid(frames[b, :, -1], nrow=5)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
  except KeyboardInterrupt:
    sys.exit()


if __name__ == "__main__":
  app.run(main)
