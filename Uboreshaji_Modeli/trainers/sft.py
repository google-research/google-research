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

"""SFT Trainer Strategy for composed generative models (VLM, Text, Audio)."""

from collections.abc import Callable, Mapping
import datetime
import json
import os
import pathlib
from typing import Any

from absl import logging
import datasets
import ml_collections
import torch
import torch.distributed as dist
from torch.utils import data as torch_data
import transformers
import trl

from Uboreshaji_Modeli.common import bagz_data
from Uboreshaji_Modeli.common import distributed_training
from Uboreshaji_Modeli.common import sft_trainer
from Uboreshaji_Modeli.engines import base as engines_base
from Uboreshaji_Modeli.trainers import base as trainers_base



def is_global_master():
  """Returns True if the current process is the global master (rank 0)."""
  return not dist.is_initialized() or dist.get_rank() == 0


def _on_rank0(fn):
  """Runs `fn` on rank 0 only and returns its result on every rank."""
  result = [fn() if is_global_master() else None]
  if dist.is_initialized():
    dist.broadcast_object_list(result, src=0)
  return result[0]


class GemmaSFTTrainerStrategy(trainers_base.TrainerStrategy):
  """Universal SFT Trainer Strategy for composed generative tasks."""

  def train(
      self,
      engine,
      dataset,
      cfg,
      **kwargs,
  ):
    """Executes SFT training.

    Args:
      engine: The model engine to use for training.
      dataset: The dataset to train on.
      cfg: The training configuration.
      **kwargs: Additional keyword arguments. Requires 'device', 'output_path',
        'model', and 'processor'.
    """
    del self  # This method does not depend on instance state.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
      if not hasattr(cfg, "run_timestamp") or not cfg.run_timestamp:
        cfg.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
      cfg.run_timestamp = ""

    # Distributed process groups are initialized upstream by
    # main_lib.run_training() before invoking the trainer strategy.

    try:
      if dist.is_initialized():
        objects = [cfg.run_timestamp]
        dist.broadcast_object_list(objects, src=0)
        cfg.run_timestamp = objects[0]

      device = kwargs.get("device")
      output_path = kwargs.get("output_path")
      if output_path is not None:
        output_path = pathlib.Path(output_path)
      model = kwargs.get("model")
      processor = kwargs.get("processor")

      if (
          model is None
          or processor is None
          or device is None
          or output_path is None
      ):
        raise ValueError(
            "model, processor, device, and output_path are required in kwargs."
        )

      # Enforce bf16 vs fp32 mapping.
      raw_precision = cfg.training.precision
      if hasattr(raw_precision, "value"):
        requested_precision = raw_precision.value
      else:
        requested_precision = str(raw_precision).lower()
      if "." in requested_precision:
        requested_precision = requested_precision.rsplit(".", 1)[-1].lower()
      device_type = getattr(device, "type", str(device))
      if requested_precision == "bf16" and not (
          torch.cuda.is_available() or device_type == "tpu"
      ):
        requested_precision = "fp32"
      elif requested_precision == "fp16" and device_type == "tpu":
        raise ValueError(
            "FP16 precision is not supported on TPU. Use BF16 or FP32."
        )

      # Apply vision tower freezing if configured. Delegates to the
      # ModelEngine helper.
      if cfg.training.get("freeze_vision_tower", False):
        engine.freeze_vision_tower(model)
        if local_rank == 0:
          logging.info("Froze vision tower parameters.")

      is_bagz = cfg.dataset.get("format") == "bagz"
      if not is_bagz:
        # Composed engines expect dataset_features to be passed in kwargs.
        train_split = dataset[cfg.dataset.train_split]
        transform_fn = engine.get_transform_fn(
            processor=processor,
            text_inputs=[],
            dataset_id2label=[],
            model_label2id={},
            cfg=cfg,
            is_train=True,
            dataset_features=train_split.features,
        )
        if isinstance(train_split, datasets.IterableDataset):
          train_dataset = train_split.map(transform_fn, batched=True)
        else:
          train_dataset = train_split.with_transform(transform_fn)

      original_output_path = output_path
      # Rank 0 aligns output directory.
      output_path = original_output_path / f"run_{cfg.run_timestamp}"

      if is_global_master():
        dir_created = False
        if not dir_created:
          output_path.mkdir(parents=True, exist_ok=True)
        logging.info("Created output directory: %r", output_path)

        config_save_path = output_path / "config.json"
        config_json = json.dumps(cfg.to_dict(), indent=2, default=str)
        config_written = False
        if not config_written:
          config_save_path.write_text(config_json)

      if dist.is_initialized():
        dist.barrier()

      training_output_dir = str(output_path)
      callbacks = []
      save_total_limit = cfg.training.get("save_total_limit", 2)

      resume_setting = cfg.training.get("resume_from_checkpoint", "auto")
      resume_setting_str = str(resume_setting).strip().lower()
      if resume_setting_str in ("auto", "true"):
        resume_from_checkpoint = None
        if resume_from_checkpoint is None:
          existing_checkpoints = list(output_path.glob("checkpoint-*"))
          resume_from_checkpoint = bool(existing_checkpoints)
      elif resume_setting_str in ("false", "none", ""):
        resume_from_checkpoint = False
      else:
        resume_from_checkpoint = resume_setting

      # Load SFT configuration overrides from composed preprocessor
      sft_overrides = engine.get_sft_config_overrides(cfg)
      default_optim = "adamw_torch" if device_type == "tpu" else None
      optim = cfg.training.get("optim", default_optim)
      sft_config_args = dict(
          output_dir=training_output_dir,
          ddp_find_unused_parameters=cfg.training.get(
              "ddp_find_unused_parameters", True
          ),
          per_device_train_batch_size=cfg.training.batch_size,
          num_train_epochs=cfg.training.num_train_epochs,
          max_steps=cfg.training.max_steps,
          bf16=(requested_precision == "bf16"),
          fp16=(requested_precision == "fp16"),
          save_strategy=cfg.training.get("save_strategy", "steps"),
          save_steps=cfg.training.get("save_steps", 200),
          save_total_limit=save_total_limit,
          logging_steps=cfg.training.logging_steps,
          learning_rate=cfg.training.learning_rate,
          weight_decay=cfg.training.weight_decay,
          remove_unused_columns=False,
          push_to_hub=False,
          dataloader_pin_memory=False,
          dataloader_num_workers=cfg.training.get(
              "dataloader_num_workers",
              0,
          ),
          gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
          report_to="tensorboard",
          seed=cfg.training.seed,
          data_seed=cfg.training.data_seed,
          gradient_checkpointing=cfg.training.gradient_checkpointing,
          logging_first_step=True,
          accelerator_config={"dispatch_batches": False},
          **sft_overrides,
      )
      if optim is not None:
        sft_config_args["optim"] = optim
      run_eval = is_bagz and cfg.dataset.get("eval_num_records")
      if is_bagz:
        # The stream resumes itself; stop TRL peeking at the first batch.
        sft_config_args.update(
            ignore_data_skip=True, completion_only_loss=False
        )
      if run_eval:
        # Loss only (no compute_metrics); WER is a separate post-training job.
        sft_config_args.update(
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            per_device_eval_batch_size=cfg.eval.eval_batch_size,
        )
      sft_config = trl.SFTConfig(**sft_config_args)

      pad_token_id = (
          processor.tokenizer.pad_token_id
          if hasattr(processor, "tokenizer")
          else 0
      )
      collate_fn = engine.get_collate_fn(
          cfg, processor=processor, pad_token_id=pad_token_id
      )
      eval_dataset = None
      if is_bagz:
        start_step = 0
        if resume_from_checkpoint:
          start_step = int(
              str(resume_from_checkpoint).rstrip("/").rsplit("-", 1)[1]
          )
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        train_dataset = bagz_data.TrainStream(
            dataset,
            collate_fn,
            seed=cfg.training.data_seed,
            batch_size=cfg.training.batch_size,
            grad_accum=cfg.training.gradient_accumulation_steps,
            start_step=start_step,
            rank=rank,
            world_size=world_size,
        )
        callbacks.append(train_dataset)
      if run_eval:
        eval_dataset = bagz_data.eval_stream(
            bagz_data.split_source(cfg, cfg.dataset.eval_split),
            collate_fn,
            seed=cfg.training.data_seed,
            num_records=cfg.dataset.eval_num_records,
            batch_size=cfg.eval.eval_batch_size,
            rank=rank,
            world_size=world_size,
        )

      is_ctc = hasattr(engine, "is_ctc") and engine.is_ctc
      is_encoder_decoder = hasattr(model, "config") and getattr(
          model.config, "is_encoder_decoder", False
      )

      if is_ctc or is_encoder_decoder:
        if local_rank == 0:
          logging.info(
              "Initializing standard transformers.Trainer for CTC or"
              " Encoder-Decoder task..."
          )

        # Apply quantization delegation patch if needed
        #    (mirrors CustomSFTTrainer behavior).
        if model is not None:
          if not hasattr(model, "hf_quantizer") and hasattr(
              model, "get_base_model"
          ):
            base_model = model.get_base_model()
            if hasattr(base_model, "hf_quantizer"):
              model.hf_quantizer = base_model.hf_quantizer

        sft = transformers.Trainer(
            model=model,
            args=sft_config,
            data_collator=collate_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            callbacks=callbacks,
        )
      else:
        if local_rank == 0:
          logging.info("Initializing CustomSFTTrainer...")
        sft = sft_trainer.CustomSFTTrainer(
            model=model,
            args=sft_config,
            data_collator=collate_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            peft_config=None,  # PEFT/LoRA is pre-wrapped in ModelEngine loader.
            callbacks=callbacks,
        )
      if is_bagz:
        # Batches are already collated and per-rank; accelerate must not
        # re-shard them.
        sft.get_train_dataloader = lambda: torch_data.DataLoader(
            train_dataset, batch_size=None
        )
        sft.get_eval_dataloader = lambda _=None: eval_dataset

      if local_rank == 0:
        logging.info(
            "SFTTrainer initialized. Starting training"
            " (resume_from_checkpoint=%s)...",
            resume_from_checkpoint,
        )

      sft.train(resume_from_checkpoint=resume_from_checkpoint)

      if local_rank == 0:
        logging.info("SFT training completed successfully.")


    finally:
      distributed_training.cleanup_distributed()
