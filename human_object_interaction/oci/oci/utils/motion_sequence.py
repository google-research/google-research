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

"""Motion Sequence generator."""
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=dangerous-default-value
# pylint: disable=unused-import
# pylint: disable=g-multiple-import
import numpy as np
from oci.utils import smpl_utils
import smplx
import torch


class MotionSequenceGenerator:

  def __init__(self, opts, model, smpl_model_path):
    self.smpl_model_path = smpl_model_path
    self.num_steps = opts.num_future_frames
    self.model = model

    return

  def generate_sequence(self, input_data, batch_index=0):
    smpl_keys = ["smpl_pose", "smpl_transl", "smpl_beta"]
    with torch.no_grad():
      model_inputs = {}
      for key in smpl_keys:
        model_inputs[key] = input_data[key] * 1
      model_inputs["gender"] = input_data["gender"]

      sequence_predictions = {}
      for key in model_inputs.keys():
        sequence_predictions[key] = []

      for i in range(self.num_steps):
        predictions = self.model.forward(model_inputs)
        for key in smpl_keys:
          model_inputs[key] = torch.cat(
              [model_inputs[key], predictions[key][:, None, :]], axis=1)[:, 1:]
          sequence_predictions[key].append(predictions[key])

      for key in smpl_keys:
        sequence_predictions[key] = torch.stack(
            sequence_predictions[key], axis=1)
        sequence_predictions[key] = torch.cat(
            [input_data[key], sequence_predictions[key]], axis=1)

      gender_id = model_inputs["gender"][batch_index].item()
      if gender_id == 1:
        gender = "male"
      elif gender_id == 2:
        gender = "female"

      meshes = smpl_utils.convert_smpl_seq2mesh(
          smpl_model_path=self.smpl_model_path,
          _pose=sequence_predictions["smpl_pose"][batch_index],
          _transl=sequence_predictions["smpl_transl"][batch_index],
          _beta=sequence_predictions["smpl_beta"][batch_index],
          gender=gender,
      )

    return meshes, sequence_predictions
