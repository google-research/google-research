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

"""Launch script for pre-training representations."""

import os.path as osp
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import torch
from xirl import common
from base_configs import validate_config
from utils import setup_experiment
from torchkit import CheckpointManager
from torchkit import experiment
from torchkit import Logger
from torchkit.utils.py_utils import Stopwatch
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_boolean("resume", False, "Whether to resume training.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_boolean("raw_imagenet", False, "")

config_flags.DEFINE_config_file(
    "config",
    "base_configs/pretrain.py",
    "File path to the training hyperparameter configuration.",
)


@experiment.pdb_fallback
def main(_):
  # Make sure we have a valid config that inherits all the keys defined in the
  # base config.
  validate_config(FLAGS.config, mode="pretrain")

  config = FLAGS.config
  exp_dir = osp.join(config.root_dir, FLAGS.experiment_name)
  setup_experiment(exp_dir, config, FLAGS.resume)

  # No need to do any pretraining if we're loading the raw pretrained
  # ImageNet baseline.
  if FLAGS.raw_imagenet:
    return

  # Setup compute device.
  if torch.cuda.is_available():
    device = torch.device(FLAGS.device)
  else:
    logging.info("No GPU device found. Falling back to CPU.")
    device = torch.device("cpu")
  logging.info(f"Using device: {device}")

  # Set RNG seeds.
  if config.seed is not None:
    logging.info(f"Pretraining experiment seed: {config.seed}")
    experiment.seed_rngs(config.seed)
    experiment.set_cudnn(config.cudnn_deterministic, config.cudnn_benchmark)
  else:
    logging.info("No RNG seed has been set for this pretraining experiment.")

  logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)

  # Load factories.
  (
      model,
      optimizer,
      pretrain_loaders,
      downstream_loaders,
      trainer,
      eval_manager,
  ) = common.get_factories(config, device)

  # Create checkpoint manager.
  checkpoint_dir = osp.join(exp_dir, "checkpoints")
  checkpoint_manager = CheckpointManager(
      checkpoint_dir,
      model=model,
      optimizer=optimizer,
  )

  global_step = checkpoint_manager.restore_or_initialize()
  total_batches = max(1, len(pretrain_loaders["train"]))
  epoch = int(global_step / total_batches)
  complete = False
  stopwatch = Stopwatch()
  try:
    while not complete:
      logger.log_learning_rate(optimizer, global_step, "pretrain")
      for batch in pretrain_loaders["train"]:
        train_loss = trainer.train_one_iter(batch)

        if not global_step % config.logging_frequency:
          for k, v in train_loss.items():
            logger.log_scalar(v, global_step, k, "pretrain")
          logger.flush()

        if not global_step % config.eval.eval_frequency:
          # Evaluate the model on the pretraining validation dataset.
          valid_loss = trainer.eval_num_iters(
              pretrain_loaders["valid"],
              config.eval.val_iters,
          )
          for k, v in valid_loss.items():
            logger.log_scalar(v, global_step, k, "pretrain")

          # Evaluate the model on the downstream datasets.
          for split, downstream_loader in downstream_loaders.items():
            eval_to_metric = eval_manager.evaluate(
                model,
                downstream_loader,
                device,
                config.eval.val_iters,
            )
            for eval_name, eval_out in eval_to_metric.items():
              eval_out.log(
                  logger,
                  global_step,
                  eval_name,
                  f"downstream/{split}",
              )

        # Save model checkpoint.
        if not global_step % config.checkpointing_frequency:
          checkpoint_manager.save(global_step)

        # Exit if complete.
        global_step += 1
        if global_step > config.optim.train_max_iters:
          complete = True
          break

        time_per_iter = stopwatch.elapsed()
        logging.info(
            "Iter[{}/{}] (Epoch {}), {:.1f}s/iter, Loss: {:.3f}".format(
                global_step,
                config.optim.train_max_iters,
                epoch,
                time_per_iter,
                train_loss["train/total_loss"].item(),
            ))
        stopwatch.reset()
      epoch += 1

  except KeyboardInterrupt:
    logging.info("Caught keyboard interrupt. Saving model before quitting.")

  finally:
    checkpoint_manager.save(global_step)
    logger.close()


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_name")
  app.run(main)
