# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Script contating function to build dataset."""

from light_field_neural_rendering.src.datasets import ff_epipolar
from light_field_neural_rendering.src.datasets import forward_facing

dataset_dict = {
    "forward_facing": forward_facing.ForwardFacing,
    "ff_epipolar": ff_epipolar.FFEpipolar,
}


def create_dataset(args):
  """Function to create dataset."""

  train_ds = dataset_dict[args.dataset.name]("train", args)
  if dataset_is_epipolar(args.dataset.name):
    # For `epipolar` dataset we need to supply train images to the eval dataset.
    eval_ds = dataset_dict[args.dataset.name]("test", args, train_ds)
  else:
    eval_ds = dataset_dict[args.dataset.name]("test", args)

  # Set config for data stats
  args.dataset.num_train_views = train_ds.size

  return train_ds, eval_ds


def dataset_is_epipolar(dataset_name):
  """Check if dataset is for epipolar model."""
  if dataset_name in ["ff_epipolar", "blender_epipolar"]:
    return True
  else:
    return False
