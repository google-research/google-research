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

"""Runs imports necessary to register Gin configurables."""

# pylint: disable=unused-import
import gin
import gin.tf
import gin.tf.external_configurables

from tf3d import data_provider
from tf3d import standard_fields
from tf3d.instance_segmentation import metric as instance_segmentation_metric
from tf3d.instance_segmentation import model as instance_segmentation_model
from tf3d.losses import box_prediction_losses
from tf3d.losses import classification_losses
from tf3d.losses import metric_learning_losses
from tf3d.object_detection import data_preparation_utils
from tf3d.object_detection import metric as object_detection_metric
from tf3d.object_detection import model as object_detection_model
from tf3d.object_detection import preprocessor as object_detection_preprocessor
from tf3d.semantic_segmentation import metric as semantic_segmentation_metric
from tf3d.semantic_segmentation import model as semantic_segmentation_model
from tf3d.semantic_segmentation import preprocessor as semantic_segmentation_preprocessor
from tf3d.utils import callback_utils
# pylint: enable=unused-import
