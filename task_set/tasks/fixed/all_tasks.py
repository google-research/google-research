# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# python3
# pylint: disable=unused-import
"""Load all fixed tasks. Each import registers some tasks."""
import task_set.tasks.fixed.fixed_2d
import task_set.tasks.fixed.fixed_image_conv
import task_set.tasks.fixed.fixed_image_conv_ae
import task_set.tasks.fixed.fixed_image_conv_vae
import task_set.tasks.fixed.fixed_language_modeling
import task_set.tasks.fixed.fixed_maf
import task_set.tasks.fixed.fixed_mlp
import task_set.tasks.fixed.fixed_mlp_ae
import task_set.tasks.fixed.fixed_mlp_vae
import task_set.tasks.fixed.fixed_nvp
import task_set.tasks.fixed.fixed_text_rnn_classification
