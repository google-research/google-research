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

"""Universal, modality-agnostic orchestration for Uboreshaji Modeli SFT."""




def run_training(
    cfg,
    *,
    output_path,
    engine_factory=engines_factory.get_engine,
    dataset_getter=data.get_dataset,
    trainer_factory=trainers_factory.get_trainer,
    now_fn=datetime.datetime.now,
    **kwargs,
):
  """Coordinates multimodal model fine-tuning using composition strategies.

  Args:
    cfg: A ConfigDict containing the configuration for the training run,
      including model flavor, model ID, task type, and dataset parameters.
    output_path: The directory where training outputs (e.g., checkpoints, logs)
      will be saved.
    engine_factory: Callable to get the ModelEngine coordinator. Defaults to
      `engines_factory.get_engine`.
    dataset_getter: Callable to load the dataset. Defaults to
      `data.get_dataset`.
    trainer_factory: Callable to get the TrainerStrategy. Defaults to
      `trainers_factory.get_trainer`.
    now_fn: Callable to get the current datetime. Defaults to
      `datetime.datetime.now`.
    **kwargs: Additional keyword arguments. These are passed to the
      `engine.load_model_and_processor` and `strategy.train` methods, containing
      parameters required by specific ModelEngine or TrainerStrategy
      implementations.

  Raises:
    FileNotFoundError: If the specified model path does not exist on CNS.
    TimeoutError: If the local rank 0 fails to copy model weights within the
      timeout period.
    RuntimeError: If the file copy command fails for any reason.
  """
  logging.info("Starting orchestrator for model flavor: %s", cfg.model_flavor)

  target_device = cfg.training.get("device", "cuda")
  if target_device == "cpu":
    device = torch.device("cpu")
    logging.info("Explicitly using CPU as requested in Config.")
  elif not torch.accelerator.is_available():
    device = torch.device("cpu")
    logging.info("No accelerator found, defaulting to CPU fallback.")
  else:
    try:
      device = torch.accelerator.current_accelerator()
      if device is None:
        raise RuntimeError("No dynamic accelerator detected.")
      logging.info("Dynamic accelerator detected: %s", device)
    except (RuntimeError, ValueError) as e:
      # Catch specific PyTorch backend errors or ValueError.
      if target_device in ["tpu", "xla"]:
        device = torch.device("xla")
        logging.info(
            "Fallback to standard XLA convention for TPU due to: %s", e
        )
      else:
        # If it failed on a GPU or other device, let it fail fast.
        raise


  engine = engine_factory(cfg.model_flavor)


  logging.info("Loading model and processor from: %s", model_id)
  model, processor = engine.load_model_and_processor(
      model_id, device, cfg=cfg, output_path=output_path, **kwargs
  )

  logging.info("Loading dataset...")
  dataset = dataset_getter(cfg)

  logging.info("Resolving trainer strategy for task type: %s", cfg.task_type)
  strategy = trainer_factory(cfg.task_type)

  logging.info("Launching strategy training loop...")
  strategy.train(
      engine,
      dataset,  # pyrefly: ignore[bad-argument-type]
      cfg,
      model=model,
      processor=processor,
      device=device,
      output_path=output_path,
      **kwargs,
  )
  logging.info("Training strategy completed successfully.")
