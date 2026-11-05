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

"""Test utilities."""

import os
import sys
import threading
from typing import Iterable, List, Optional, Union

from absl import logging
import tensorflow as tf

tpu_initialized = None
tpu_initialized_lock = threading.Lock()


class TestCase(tf.test.TestCase):
  """Test case which handles hard device placement."""

  ENTER_PRIMARY_DEVICE = True

  def setUp(self):
    super().setUp()

    # Enable autograph strict mode - any autograph errors will trigger an error
    # rather than falling back to no conversion.
    os.environ["AUTOGRAPH_STRICT_CONVERSION"] = "1"

    self._device_types = frozenset(
        d.device_type for d in tf.config.experimental.list_logical_devices())
    self.on_tpu = "TPU" in self._device_types
    logging.info("Physical devices: %s", tf.config.list_physical_devices())
    logging.info("Logical devices: %s", tf.config.list_logical_devices())

    # Initialize the TPU system once and only once.
    global tpu_initialized
    if tpu_initialized is None:
      with tpu_initialized_lock:
        if tpu_initialized is None and self.on_tpu:
          tf.tpu.experimental.initialize_tpu_system()
        tpu_initialized = True

    if self.ENTER_PRIMARY_DEVICE:
      self._device = tf.device("/device:%s:0" % self.primary_device)
      self._device.__enter__()

  def tearDown(self):
    super().tearDown()
    if self.ENTER_PRIMARY_DEVICE:
      self._device.__exit__(*sys.exc_info())
      del self._device

  @property
  def primary_device(self):
    if "TPU" in self._device_types:
      return "TPU"
    elif "GPU" in self._device_types:
      return "GPU"
    else:
      return "CPU"


def simulate_two_devices(device_types = ("GPU", "CPU")):
  """Splits the primary device (CPU/GPU) into two virtual devices.

  Call this inside the setUpModule() before TF has the chance to initialize
  the primary device.
  It will first try splitting the first GPU (if available). If no GPU is
  available it will split the first CPU device.

  Args:
    device_types: List of devices for which should be split (if available).
  """
  for device_type in device_types:
    devices = tf.config.list_physical_devices(device_type=device_type)
    if not devices:
      continue
    assert len(devices) == 1, devices
    # For GPUs we should adjust the memory limit. GPUs on Forge have 16 GB,
    # but 2 * 4 GB should be enough for tests.
    memory_limit = 4096 if device_type == "GPU" else None  # MB
    tf.config.experimental.set_virtual_device_configuration(
        devices[0], [
            tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=memory_limit),
            tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=memory_limit)
        ])
    logging.info("Split %s into two virtual devices with memory limit %s.",
                 devices[0], memory_limit)
    return
  logging.error(
      "Did not split any device into two device, physical devices: %s",
      tf.config.list_physical_devices())


def _get_gpu_or_cpu_devices():
  """Get the list of GPU or CPU devicen (prefering GPUs)."""
  for device_type in ["GPU", "CPU"]:
    devices = tf.config.experimental.list_logical_devices(
        device_type=device_type)
    if devices:
      return [d.name for d in devices]
  raise ValueError(
      "Could not find any logical devices of type GPU or CPU, logical devices: "
      f"{tf.config.experimental.list_logical_devices()}")


DistributionStrategy = Union[tf.distribute.Strategy,
                             tf.distribute.OneDeviceStrategy,
                             tf.distribute.MirroredStrategy,
                             tf.distribute.experimental.TPUStrategy]


def create_distribution_strategy(use_tpu):
  """Create a distribution strategy.

  Args:
   use_tpu: Uses a TPU strategy.

  Returns:
    Distribution strategy.
  """
  if use_tpu:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    devices = _get_gpu_or_cpu_devices()
    if len(devices) == 1:  # pyrefly: ignore[bad-argument-type]
      strategy = tf.distribute.OneDeviceStrategy(devices[0])  # pyrefly: ignore[unsupported-operation]
    else:
      strategy = tf.distribute.MirroredStrategy(devices)

  logging.info("Devices: %s", tf.config.list_logical_devices())
  logging.info("Distribution strategy: %s", strategy)
  return strategy
