# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Trainer strategy for object detection models."""

from collections.abc import Mapping
import datetime
import json
import os
import time
from typing import Any

from absl import logging
from etils import epath
import ml_collections
import torch  # pylint: disable=unused-import
import torch.distributed as dist
import transformers

from Uboreshaji_Modeli.common import metrics
from Uboreshaji_Modeli.common import trainer
from Uboreshaji_Modeli.engines import base


def _is_global_master() -> bool:
  """Returns True if the current process is the global master (rank 0)."""
  return not dist.is_initialized() or dist.get_rank() == 0


class DetectionTrainer(trainers_base.TrainerStrategy):
  """Strategy for training object detection models using composed components."""

  def train(
      self,
      engine: base.ModelEngine,
      dataset: Mapping[str, Any],
      cfg: ml_collections.ConfigDict,
      **kwargs,
  ) -> None:
    """Executes the composed OWL-v2 training and evaluation loop.

    Args:
      engine: The model engine to use for training.
      dataset: The dataset containing 'train', 'valid', and optionally 'test'
        splits.
      cfg: The configuration for training and evaluation.
      **kwargs: Additional keyword arguments. Requires 'device' (the device to
        run on), 'output_path' (the directory to save outputs), 'model' (the
        model to train), and 'processor' (the model's preprocessor).
    """
    device = kwargs.get("device")
    if device is None:
      raise ValueError("device is required in kwargs for DetectionTrainer.")

    output_path = kwargs.get("output_path")
    if output_path is None:
      raise ValueError(
          "output_path is required in kwargs for DetectionTrainer."
      )
    output_path = epath.Path(output_path)

    if not hasattr(cfg, "run_timestamp") or not cfg.run_timestamp:
      cfg.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    is_cns = str(output_path).startswith("/cns")
    if is_cns:
      resolved_output_path = output_path
    else:
      resolved_output_path = output_path / f"run_{cfg.run_timestamp}"
      # In distributed training, only master rank creates directory
      local_rank = int(os.environ.get("LOCAL_RANK", 0))
      if local_rank == 0:
        resolved_output_path.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
      config_save_path = resolved_output_path / "config.json"
      saved_internally = False


      if not saved_internally:
        config_save_path.write_text(
            json.dumps(cfg.to_dict(), indent=2, default=str)
        )

    model = kwargs.get("model")
    processor = kwargs.get("processor")
    if model is None or processor is None:
      raise ValueError(
          "model and processor are required in kwargs for DetectionTrainer."
      )

    train_dataset = dataset[cfg.dataset.train_split]
    eval_dataset = dataset[cfg.dataset.eval_split]
    test_split_name = cfg.eval.get(
        "split", cfg.dataset.get("test_split", "test")
    )
    test_dataset = dataset.get(test_split_name, eval_dataset)

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=str(resolved_output_path),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.eval.eval_batch_size,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        report_to="tensorboard",
        logging_dir=str(resolved_output_path),
        seed=cfg.training.seed,
        data_seed=cfg.training.data_seed,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        metric_for_best_model="eval_map_50",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        max_grad_norm=cfg.training.max_grad_norm,
    )

    compute_metrics_fn = metrics.create_compute_metrics_fn(
        resize_to=cfg.dataset.image_size, score_threshold=0.0
    )

    label_by_dataset_id = train_dataset.features["objects"][
        "category"
    ].feature.names
    categories_to_exclude = set(cfg.dataset.get("exclude_classes", []))
    valid_categories = [
        name for name in label_by_dataset_id
        if name not in categories_to_exclude
    ]
    num_classes = len(valid_categories)

    id_by_model_label = {name: i for i, name in enumerate(valid_categories)}
    transform_fn = engine.get_transform_fn(
        processor=processor,
        text_inputs=valid_categories,
        dataset_id2label=label_by_dataset_id,
        model_label2id=id_by_model_label,
        cfg=cfg,
        is_train=True,
    )
    eval_transform_fn = engine.get_transform_fn(
        processor=processor,
        text_inputs=valid_categories,
        dataset_id2label=label_by_dataset_id,
        model_label2id=id_by_model_label,
        cfg=cfg,
        is_train=False,
    )

    transformed_train_dataset = train_dataset.with_transform(transform_fn)
    transformed_eval_dataset = eval_dataset.with_transform(eval_transform_fn)
    transformed_test_dataset = None
    if test_dataset is not None:
      transformed_test_dataset = test_dataset.with_transform(eval_transform_fn)

    criterion, weight_dict = engine.get_criterion(num_classes, cfg, device)
    collate_fn = engine.get_collate_fn(cfg)

    callbacks = []

      custom_trainer = trainer.CustomTrainer(
          model=model,
          args=training_args,
          data_collator=collate_fn,
          train_dataset=transformed_train_dataset,
          eval_dataset=transformed_eval_dataset,
          compute_metrics=compute_metrics_fn,
          processing_class=processor,
          criterion=criterion,
          weight_dict=weight_dict,
          callbacks=callbacks,
      )

      start_time = time.monotonic()
      if not cfg.eval.get("run_eval_only", False):
        logging.info("Starting training...")
        custom_trainer.train(resume_from_checkpoint=False)

      else:
        logging.info("Skipping training as run_eval_only is set to True.")

      logging.info("Running final evaluation...")
      eval_results = custom_trainer.evaluate(
          eval_dataset=transformed_test_dataset, metric_key_prefix="best_eval"
      )
      logging.info("Final eval results: %s", eval_results)

      train_loss = None
      for entry in reversed(custom_trainer.state.log_history):
        if "loss" in entry:
          train_loss = entry["loss"]
          break

      train_metrics = {
          "wall_clock_seconds": time.monotonic() - start_time,
          "final_training_loss": train_loss,
          **eval_results,
      }

      if _is_global_master():
        eval_json_path = resolved_output_path / "evaluation.json"
        eval_json_path.write_text(json.dumps(train_metrics, indent=2))
        logging.info("Saved evaluation.json to %s", eval_json_path)

      logging.info("Final training metrics exported: %s", train_metrics)
