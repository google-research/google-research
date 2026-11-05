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

"""Tests for distributed_training module."""

import datetime
import os
import unittest

from absl.testing import absltest
import torch
import torch.distributed as dist

from Uboreshaji_Modeli.common import distributed_training


class InitDistributedTest(absltest.TestCase):
  """Tests for init_distributed()."""

  def setUp(self):
    """Saves environment variables modified by init_distributed."""
    super().setUp()
    self._saved_env = {}
    for key in [
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "PYTORCH_CUDA_ALLOC_CONF",
        "IS_TPU",
        "BORG_CELL",
        "MASTER_ADDR",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    ]:
      self._saved_env[key] = os.environ.get(key)

  def tearDown(self):
    """Restores environment variables and destroys any active process group."""
    for key, val in self._saved_env.items():
      if val is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = val
    if dist.is_initialized():
      dist.destroy_process_group()
    super().tearDown()

  def test_single_device_sets_env_and_skips_process_group(self):
    """Without WORLD_SIZE set, should configure CUDA but skip process group."""
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("LOCAL_RANK", None)
    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=True, autospec=True
        )
    )
    self.enter_context(
        unittest.mock.patch.object(torch.cuda, "set_device", autospec=True)
    )

    distributed_training.init_distributed(seed=42)

    self.assertEqual(
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True"
    )
    self.assertFalse(dist.is_initialized())

  def test_world_size_one_skips_process_group(self):
    """With WORLD_SIZE=1, should skip process group initialization."""
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=True, autospec=True
        )
    )
    self.enter_context(
        unittest.mock.patch.object(torch.cuda, "set_device", autospec=True)
    )

    distributed_training.init_distributed(seed=42)

    self.assertFalse(dist.is_initialized())

  @unittest.mock.patch.object(
      torch.cuda, "is_available", return_value=False, autospec=True
  )
  def test_no_cuda_skips_device_setup(self, mock_cuda):
    """Without CUDA, should skip device setup and process group."""
    os.environ.pop("WORLD_SIZE", None)

    distributed_training.init_distributed(seed=42)

    self.assertFalse(dist.is_initialized())
    mock_cuda.assert_called()

  def test_world_size_greater_than_one_calls_init(self):
    """With WORLD_SIZE>1, should init process group with NCCL and default timeout."""
    os.environ["WORLD_SIZE"] = "8"
    os.environ["LOCAL_RANK"] = "3"
    os.environ["LOCAL_WORLD_SIZE"] = "8"
    os.environ["GROUP_RANK"] = "0"

    mock_init = self.enter_context(
        unittest.mock.patch.object(dist, "init_process_group", autospec=True)
    )
    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=True, autospec=True
        )
    )
    mock_set_device = self.enter_context(
        unittest.mock.patch.object(torch.cuda, "set_device", autospec=True)
    )

    distributed_training.init_distributed(seed=42)

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args
    self.assertEqual(call_kwargs.kwargs["backend"], "NCCL")
    self.assertEqual(call_kwargs.kwargs["rank"], 3)
    self.assertEqual(call_kwargs.kwargs["world_size"], 8)
    self.assertEqual(
        call_kwargs.kwargs["timeout"],
        datetime.timedelta(minutes=180),
    )
    mock_set_device.assert_called_once()

  def test_custom_timeout(self):
    """A custom timeout should be forwarded to init_process_group."""
    os.environ["WORLD_SIZE"] = "8"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "8"
    os.environ["GROUP_RANK"] = "0"

    mock_init = self.enter_context(
        unittest.mock.patch.object(dist, "init_process_group", autospec=True)
    )
    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=True, autospec=True
        )
    )
    self.enter_context(
        unittest.mock.patch.object(torch.cuda, "set_device", autospec=True)
    )

    distributed_training.init_distributed(
        seed=42, timeout=datetime.timedelta(minutes=360)
    )

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args
    self.assertEqual(
        call_kwargs.kwargs["timeout"],
        datetime.timedelta(minutes=360),
    )

  def test_cpu_mode_sets_gloo_backend(self):
    """With no CUDA and no TPU, should use GLOO backend for world_size > 1."""
    os.environ["WORLD_SIZE"] = "4"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "4"
    os.environ["GROUP_RANK"] = "0"

    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=False, autospec=True
        )
    )
    mock_init = self.enter_context(
        unittest.mock.patch.object(dist, "init_process_group", autospec=True)
    )

    distributed_training.init_distributed(seed=42)

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args
    self.assertEqual(call_kwargs.kwargs["backend"], "GLOO")

  def test_seeding_called_for_cpu(self):
    """Should call torch.manual_seed."""
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("IS_TPU", None)
    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=False, autospec=True
        )
    )
    mock_manual_seed = self.enter_context(
        unittest.mock.patch.object(torch, "manual_seed", autospec=True)
    )

    distributed_training.init_distributed(seed=123)

    mock_manual_seed.assert_called_once_with(123)

  def test_seeding_called_for_cuda(self):
    """Should call torch.manual_seed and torch.cuda.manual_seed_all."""
    os.environ.pop("WORLD_SIZE", None)
    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=True, autospec=True
        )
    )
    self.enter_context(
        unittest.mock.patch.object(torch.cuda, "set_device", autospec=True)
    )
    mock_manual_seed = self.enter_context(
        unittest.mock.patch.object(torch, "manual_seed", autospec=True)
    )
    mock_cuda_manual_seed = self.enter_context(
        unittest.mock.patch.object(torch.cuda, "manual_seed_all", autospec=True)
    )

    distributed_training.init_distributed(seed=123)

    mock_manual_seed.assert_called_once_with(123)
    mock_cuda_manual_seed.assert_called_once_with(123)

  def test_multi_node_rank_computation(self):
    """Global rank should be computed as node_rank * local_world_size + local_rank."""
    os.environ["WORLD_SIZE"] = "16"
    os.environ["LOCAL_RANK"] = "2"
    os.environ["LOCAL_WORLD_SIZE"] = "8"
    os.environ["GROUP_RANK"] = "1"

    mock_init = self.enter_context(
        unittest.mock.patch.object(dist, "init_process_group", autospec=True)
    )
    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=True, autospec=True
        )
    )
    mock_set_device = self.enter_context(
        unittest.mock.patch.object(torch.cuda, "set_device", autospec=True)
    )

    distributed_training.init_distributed(seed=42)

    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args
    self.assertEqual(call_kwargs.kwargs["backend"], "NCCL")
    self.assertEqual(call_kwargs.kwargs["rank"], 10)
    self.assertEqual(call_kwargs.kwargs["world_size"], 16)
    mock_set_device.assert_called_once()



class CleanupDistributedTest(absltest.TestCase):
  """Tests for cleanup_distributed()."""

  @unittest.mock.patch.object(
      dist, "is_initialized", return_value=False, autospec=True
  )
  def test_noop_when_not_initialized(self, unused_mock):
    """Should not call destroy_process_group when no group is active."""
    with unittest.mock.patch.object(
        dist, "destroy_process_group", autospec=True
    ) as mock_destroy:
      distributed_training.cleanup_distributed()
    mock_destroy.assert_not_called()

  @unittest.mock.patch.object(
      dist, "is_initialized", return_value=True, autospec=True
  )
  @unittest.mock.patch.object(dist, "destroy_process_group", autospec=True)
  def test_destroys_when_initialized(self, mock_destroy, unused_mock):
    """Should call destroy_process_group when a group is active."""
    distributed_training.cleanup_distributed()
    mock_destroy.assert_called_once()


if __name__ == "__main__":
  absltest.main()
