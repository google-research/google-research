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

"""Multi-task vision and language model.
"""
from typing import Any, Callable, Optional

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from modeling import base
from utils import gin_utils
from utils.task_utils import Tasks
from utils.types import Array
from utils.types import DType
from utils.types import NestedDictArray


@gin.register
class MultitaskModel(base.BaseModel):
  """Multi-task model function.

  Attributes:
    tasks: A sequence of tasks to instantiate. Each task contains its own head.
    train_vision_model: A bool specifying whether to train the vision model.
    vision_model_fn: A function returning a nn.Module specifying which vision
      encoder to use.
    dtype: A jax data type.
  """
  tasks: Tasks = gin.REQUIRED
  vision_model_fn: Callable[Ellipsis, Any] = gin.REQUIRED
  frozen_vision_model_fn: Optional[Callable[Ellipsis, Any]] = None
  train_vision_model: bool = True
  dtype: DType = jnp.float32

  def setup(self):
    """Initializes a Module lazily (similar to a lazy ``__init__``).
    """
    if not self.tasks:
      raise ValueError('Tasks must not be empty!!')

    module_attrs = {
        'train': (self.mode == base.ExecutionMode.TRAIN),
        'mode': self.mode,
        'dtype': self.dtype,
    }

    assert self.vision_model_fn is not None
    self.vision_model = self.vision_model_fn(
        **base.filter_attrs(self.vision_model_fn, module_attrs))

    if self.frozen_vision_model_fn is not None:
      self.frozen_vision_model = self.frozen_vision_model_fn(
          **base.filter_attrs(self.frozen_vision_model_fn, module_attrs))

    # Set up task heads.
    self.task_heads = [task.head(
        **base.filter_attrs(task.head, module_attrs))for task in self.tasks]

  @nn.compact
  @gin_utils.allow_remapping
  def __call__(self,
               image,
               text,
               labels):
    """Call function for the multi-task model.

    Args:
      image: An array of shape [batch_size, height, width, channels].
      text: A numeric array of the input text with shape
        [batch_size, ..., seq_len].
      labels: A dictionary with task-specific labels.

    Returns:
      model_outputs: A dictionary with task-specific outputs.
    """
    vision_features = self.vision_model(image)
    if not self.train_vision_model:
      vision_features = jax.lax.stop_gradient(vision_features)

    frozen_vision_features = None
    if self.frozen_vision_model_fn:
      frozen_vision_features = self.frozen_vision_model(image)
      frozen_vision_features = jax.lax.stop_gradient(frozen_vision_features)

    text_features = text
    image_text_features = None
    paddings = None
    model_outputs = {}
    for task, task_head in zip(self.tasks, self.task_heads):
      task_labels = task.filter_by_task(labels)
      if self.frozen_vision_model_fn is not None:
        task_outputs = task_head(vision_features, text_features,
                                 image_text_features, paddings, task_labels,
                                 frozen_vision_features)
      else:
        task_outputs = task_head(vision_features, text_features,
                                 image_text_features, paddings, task_labels)
      # Add task specific scope name.
      model_outputs.update(task.unfilter_by_task(task_outputs))

    return model_outputs
