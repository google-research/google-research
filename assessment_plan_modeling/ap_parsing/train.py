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

"""A customized training library for the AP parsing TF-NLP task."""

import os

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf

from assessment_plan_modeling.ap_parsing import ap_parsing_task
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance

FLAGS = flags.FLAGS


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir
  if "train" in FLAGS.mode:
    train_utils.serialize_config(params, model_dir)

  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu,
      **params.runtime.model_parallelism())

  with distribution_strategy.scope():
    if params.task.use_crf:
      task = ap_parsing_task.APParsingTaskCRF(params.task)
    else:
      task = ap_parsing_task.APParsingTaskBase(params.task)

    ckpt_exporter = train_utils.maybe_create_best_ckpt_exporter(
        params, model_dir)
    trainer = train_utils.create_trainer(
        params,
        task,
        train="train" in FLAGS.mode,
        evaluate=("eval" in FLAGS.mode),
        checkpoint_exporter=ckpt_exporter)

  model, _ = train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      trainer=trainer,
      model_dir=model_dir)

  train_utils.save_gin_config(FLAGS.mode, model_dir)

  # Export saved model.
  if "train" in FLAGS.mode:
    saved_model_path = os.path.join(model_dir, "saved_models/latest")
    logging.info("Exporting SavedModel to %s", saved_model_path)
    tf.saved_model.save(model, saved_model_path)

    if ckpt_exporter:
      logging.info("Loading best checkpoint for export")
      trainer.checkpoint.restore(ckpt_exporter.best_ckpt_path)
      saved_model_path = os.path.join(model_dir, "saved_models/best")

      # Make sure restored and not re-initialized.
      if trainer.global_step > 0:
        logging.info(
            "Exporting best saved model by %s (from global step: %d) to %s",
            params.trainer.best_checkpoint_eval_metric,
            trainer.global_step.numpy(), saved_model_path)
        tf.saved_model.save(trainer.model, saved_model_path)


if __name__ == "__main__":
  tfm_flags.define_flags()
  app.run(main)
