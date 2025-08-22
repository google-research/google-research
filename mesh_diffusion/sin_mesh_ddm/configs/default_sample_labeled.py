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

"""Config for training a FLDM on a labeled mesh."""
import ml_collections

from mesh_diffusion.sin_mesh_ddm.configs import default


def get_config():
  """ConfigDict example showing the params that need specifying for sampling."""
  config = default.get_config()
  config.obj_name = 'fe973fc8e8c049c6b9ab884137fc9463'

  # Directory where preprocessed meshes are stored.
  config.geom_record_path = ''  # Change me.

  # Location of the labeled ply file.
  config.obj_labels = ''  # Change me.

  # Location of the FLDM model checkpoint.
  config.model_checkpoint_dir = ''  # Change me.

  # Output directory where generated texture samples will be saved.
  config.sample_save_dir = ''  # Change me.


  config.train = False
  config.ddm_schedule = 'cos'
  config.mlp_layers = 2
  config.hdim = 16
  config.unet_features = 96
  config.sample_during_training = True
  return config

