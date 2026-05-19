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

"""Main entry point for fine-tuning experiments."""

import datetime
import json
import sys
import tempfile
import time

from absl import app
from absl import flags
from absl import logging
from etils import epath
import ml_collections
import torch
from torch.utils import tensorboard
import transformers

from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import config_utils
from Uboreshaji_Modeli.common import data
from Uboreshaji_Modeli.common import metrics
from Uboreshaji_Modeli.common import trainer
from Uboreshaji_Modeli.engines import factory


_CONFIG = flags.DEFINE_string("config", None, "Path to Python config file.")
_CONFIG_JSON = flags.DEFINE_string(
    "config_json", None, "JSON string of the experiment configuration."
)
_MODEL_ID = flags.DEFINE_string(
    "model_id", None, "Path or ID of the pretrained model (overrides config)."
)
_DATASET_PATH = flags.DEFINE_string(
    "dataset_path", None, "Path to the dataset (overrides config)."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "Directory to save outputs (overrides config)."
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name", None, "Name of the experiment (overrides config)."
)


def main(argv):
  """Runs fine-tuning experiments.

  Dispatches to the appropriate training pipeline based on the model_flavor
  configuration field.

  Args:
    argv: Command-line arguments.

  Raises:
    app.UsageError: If the config is not provided or if there are too many
      command-line arguments.
    ValueError: If the dataset type is unsupported.
    RuntimeError: If the Python version is lower than 3.11.
  """
  if sys.version_info < (3, 11):
    raise RuntimeError("This script requires Python 3.11 or higher.")

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _CONFIG_JSON.value:
    cfg = ml_collections.ConfigDict(json.loads(_CONFIG_JSON.value))
  elif _CONFIG.value:
    cfg = config_utils.load_config(_CONFIG.value)
  else:
    raise app.UsageError("Either --config or --config_json must be provided.")

  config_utils.derive_paths(cfg)

  if _MODEL_ID.value:
    cfg.model_id = _MODEL_ID.value
  if _DATASET_PATH.value:
    cfg.dataset.dataset_path = _DATASET_PATH.value
  if _OUTPUT_DIR.value:
    cfg.output_dir = _OUTPUT_DIR.value
  if _EXPERIMENT_NAME.value:
    cfg.experiment_name = _EXPERIMENT_NAME.value


  logging.info("Starting experiment: %s", cfg.experiment_name)
  logging.info("Using model: %s", cfg.model_id)
  logging.info("Using dataset: %s", cfg.dataset.dataset_path)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  use_bf16 = (cfg.training.precision == config.Precision.BF16) and (
      device.type == "cuda"
  )
  logging.info("Using device: %s", device)
  logging.info("Using bf16: %s", use_bf16)

  engine = factory.get_engine(cfg.model_flavor)
  model, processor = engine.load_model_and_processor(cfg.model_id, device)

  dataset_root = data.get_dataset(cfg)
  # Prepare class info from train split
  train_split = dataset_root[cfg.dataset.train_split]
  eval_split = dataset_root[cfg.dataset.eval_split]
  if isinstance(train_split, data.datasets.Dataset):
    # 1. Get the dataset's own CANONICAL mapping.
    #    Stable map from integer to name for this version of the dataset.
    dataset_id2label = train_split.features["objects"]["category"].feature.names

    # 2. Determine the valid class NAMES for our experiment from the config.
    categories_to_exclude = set(cfg.dataset.get("exclude_classes", []))
    valid_categories = [
        name for name in dataset_id2label if name not in categories_to_exclude
    ]
    logging.info(
        "Training with %d classes: %s", len(valid_categories), valid_categories
    )

    # 3. Create the MODEL's training map.
    #    This creates a stable, contiguous {name: id} map for this specific run.
    #    e.g., {"leaf": 0, "stem": 1, ...}
    model_label2id = {name: i for i, name in enumerate(valid_categories)}
    num_classes = len(valid_categories)
    text_inputs = valid_categories  # The model gets the filtered list of names.

    # Prepare Split and Transforms, PASSING the stable map to the transform.
    transform_fn = engine.get_transform_fn(
        processor,
        text_inputs,
        dataset_id2label,
        model_label2id,
        cfg=cfg,
        is_train=True,
    )
    eval_transform_fn = engine.get_transform_fn(
        processor,
        text_inputs,
        dataset_id2label,
        model_label2id,
        cfg=cfg,
        is_train=False,
    )
    train_dataset = train_split.with_transform(transform_fn)
    eval_dataset = eval_split.with_transform(eval_transform_fn)

    test_split = None
    test_split_name = cfg.dataset.get("test_split", "test")
    if test_split_name in dataset_root:
      logging.info("Preparing test split: %s", test_split_name)
      test_split = dataset_root[test_split_name]
      test_dataset = test_split.with_transform(eval_transform_fn)
    else:
      logging.warning(
          "Test split '%s' not found. Falling back to validation.",
          test_split_name,
      )
      test_dataset = eval_dataset

  else:
    raise ValueError(
        f"Unsupported dataset type: {type(train_split)}. Expected a HuggingFace"
        " Dataset."
    )

  criterion, weight_dict = engine.get_criterion(num_classes, cfg, device)

  original_output_path = epath.Path(cfg.output_dir)
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = original_output_path / f"run_{timestamp}"

  output_path.mkdir(parents=True, exist_ok=True)
  logging.info("Created output directory: %s", output_path)

  config_save_path = output_path / "config.json"
  config_save_path.write_text(json.dumps(cfg.to_dict(), indent=2, default=str))

  training_output_dir = str(output_path)

  training_args = transformers.TrainingArguments(
      output_dir=training_output_dir,
      per_device_train_batch_size=cfg.training.batch_size,
      per_device_eval_batch_size=1,
      num_train_epochs=cfg.training.num_train_epochs,
      bf16=use_bf16,
      logging_steps=cfg.training.logging_steps,
      learning_rate=cfg.training.learning_rate,
      weight_decay=cfg.training.weight_decay,
      remove_unused_columns=False,
      push_to_hub=False,
      dataloader_pin_memory=True,
      gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
      report_to="tensorboard",
      logging_dir=str(output_path),
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

  custom_trainer = trainer.CustomTrainer(
      model=model,
      args=training_args,
      data_collator=engine.get_collate_fn(cfg),
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=compute_metrics_fn,
      processing_class=processor,
      criterion=criterion,
      weight_dict=weight_dict,
      callbacks=[
      ],
  )

  start_time = time.monotonic()
  if not cfg.eval.get("run_eval_only", False):
    logging.info("Starting training...")
    custom_trainer.train(resume_from_checkpoint=False)

  else:
    logging.info("Skipping training as run_eval_only is set to True.")

  logging.info("Running final evaluation...")
  eval_results = custom_trainer.evaluate(eval_dataset=test_dataset)
  logging.info("Final eval results: %s", eval_results)

  train_loss = None
  for entry in reversed(custom_trainer.state.log_history):
    if "loss" in entry:
      train_loss = entry["loss"]
      break

  train_metrics = {
      "status": "COMPLETED",
      "total_steps": custom_trainer.state.global_step,
      "wall_clock_seconds": round(time.monotonic() - start_time),
  }
  if train_loss is not None:
    train_metrics["train_loss"] = train_loss

  publisher_eval = metrics.format_for_publisher(
      eval_results=eval_results,
      label_names=valid_categories,
      model_label2id=model_label2id,
      train_metrics=train_metrics,
  )
  eval_json_path = output_path / "evaluation.json"
  eval_json_path.write_text(json.dumps(publisher_eval, indent=2, default=str))
  logging.info("Saved evaluation.json to %s", eval_json_path)

  cfg.eval.eval_json = str(eval_json_path)
  config_save_path.write_text(json.dumps(cfg.to_dict(), indent=2, default=str))



if __name__ == "__main__":
  flags.mark_flags_as_mutual_exclusive(["config", "config_json"], required=True)
  app.run(main)
