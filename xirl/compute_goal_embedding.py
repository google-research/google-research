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

"""Compute and store the mean goal embedding using a trained model."""

import os
import typing
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torchkit.checkpoint import CheckpointManager
from xirl import common
from xirl.models import SelfSupervisedModel
import utils
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean(
    "restore_checkpoint", True,
    "Restore model checkpoint. Disabling loading a checkpoint is useful if you "
    "want to measure performance at random initialization.")

ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


def embed(
    model: ModelType,
    downstream_loader: DataLoaderType,
    device: torch.device,
) -> np.ndarray:
  """Embed the stored trajectories and compute mean goal embedding."""
  goal_embs = []
  for class_name, class_loader in downstream_loader.items():
    logging.debug(f"Embedding {class_name}.")
    for batch_idx, batch in enumerate(class_loader):
      if batch_idx % 100 == 0:
        logging.debug(f"\tEmbedding batch: {batch_idx}...")
      out = model.infer(batch["frames"].to(device))
      emb = out.numpy().embs
      goal_embs.append(emb[-1, :])
  goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
  return goal_emb


def setup() -> typing.Tuple[ModelType, DataLoaderType]:
  """Load the latest embedder checkpoint and dataloaders."""
  config = utils.load_config_from_dir(FLAGS.experiment_path)
  model = common.get_model(config)
  downstream_loaders = common.get_downstream_dataloaders(config, False)["train"]
  checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
  if FLAGS.restore_checkpoint:
    checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
    global_step = checkpoint_manager.restore_or_initialize()
    logging.info(f"Restored model from checkpoint {global_step}.")
  else:
    logging.info("Skipping checkpoint restore.")
  return model, downstream_loaders


def main(_):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, downstream_loader = setup()
  model.to(device).eval()
  goal_emb = embed(model, downstream_loader, device)
  utils.save_goal_embedding(FLAGS.experiment_path, goal_emb)


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_path")
  app.run(main)
