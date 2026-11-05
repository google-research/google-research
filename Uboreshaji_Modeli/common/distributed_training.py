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

"""Utilities for distributed training on multiple devices."""

import datetime
import os

from absl import logging
import torch
import torch.distributed as dist


def _resolve_device_type():
  """Resolves target execution device based on environment."""
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"




def init_distributed(
    seed,
    timeout = datetime.timedelta(minutes=180),
):
  """Initializes distributed training environment.

  Sets process-level CUDA allocator configuration (if CUDA)
  (``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``) to reduce memory
  fragmentation for large models.

  Args:
    seed: Random seed for reproducibility across all devices.
    timeout: Timeout for distributed collective operations. Defaults to 3 hours.
  """
  device_type = _resolve_device_type()

  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))

  torch.manual_seed(seed)

  if device_type == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))
    backend = "NCCL"
  else:
    backend = "GLOO"

  if world_size > 1:
    if not dist.is_initialized():
      num_workers_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
      node_rank = int(os.environ.get("GROUP_RANK", 0))
      global_rank = node_rank * num_workers_per_node + local_rank

      logging.info(
          "Initializing process group: global_rank=%d, world_size=%d,"
          " backend=%s",
          global_rank,
          world_size,
          backend,
      )
      dist.init_process_group(
          backend=backend,
          rank=global_rank,
          world_size=world_size,
          timeout=timeout,
      )
    else:
      logging.info("Process group already initialized.")
  else:
    logging.info("Skipping process group initialization for single device run.")


def cleanup_distributed():
  """Tears down the distributed process group if one is active."""
  if dist.is_initialized():
    logging.info("Destroying process group.")
    dist.destroy_process_group()
