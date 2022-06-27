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

"""Saccader-Classification network model.

Saccader model is an image classification model with a hard attention mechanism.
The model uses the saccader model for visual attention
and uses a separate network for classification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from saccader import utils
from saccader.visual_attention import saccader
from tensorflow.contrib import slim as contrib_slim
from tensorflow_models.slim.nets import nets_factory
from tensorflow_models.slim.nets.nasnet import nasnet


slim = contrib_slim
Saccader = saccader.Saccader


class SaccaderClassNet(Saccader):
  """Saccader-Classification Model.

  Network that performs classification on images by taking glimpses at
  different locations on an image.

  Attributes:
    num_classes: (Integer) Number of classification classes.
    variable_scope: (String) Name of model variable scope.
    attention_groups: (Integer) Number of groups in attention network.
    attention_layers_per_group: (Integer) Number of layers in each group in
      attention network.
    saccader_cell: Saccader Cell object.
    representation_network: Representation network object.
    glimpse_shape: 2-D tuple of integers indicating glimpse shape.
    glimpse_shape_classnet: 2-D tuple of integers indicating classification
      network glimpse shape.
    glimpse_shape_saccader: 2-D tuple of integers indicating saccader
      glimpse shape.
    var_list_representation_network: List of variables for the representation
      network.
    var_list_attention_network: List of variables for the attention network.
    var_list_saccader_cell: List of variables for the saccader cell.
    var_list_location: List of variables for the location network.
    var_list_classification: List of variables for the classification network.
    var_list_classnet: List of variables for the classification network.
    var_list: List of all model variables.
    init_op: Initialization operations for model variables.
  """

  def __init__(self, config, variable_scope="saccader_classnet"):
    Saccader.__init__(self, config, variable_scope=variable_scope+"/saccader")
    self.var_list_saccader = []
    self.var_list_classnet = []
    self.classnet_type = config.classnet_type
    self.num_classes = config.num_classes
    self.variable_scope_classnet = variable_scope+"/"+self.classnet_type
    self.glimpse_shape_saccader = (-1, -1)
    self.glimpse_shape_classnet = config.glimpse_shape

  def __call__(self,
               images_saccader,
               images_classnet,
               num_times,
               is_training_saccader=False,
               is_training_classnet=False,
               policy="learned",
               stop_gradient_after_representation=False):

    logits, locations_t, best_locations_t, endpoints = Saccader.__call__(
        self,
        images_saccader,
        num_times,
        is_training=is_training_saccader,
        policy=policy,
        stop_gradient_after_representation=stop_gradient_after_representation)

    self.glimpse_shape_saccader = self.glimpse_shape
    image_size_saccader = images_saccader.shape.as_list()[1]
    image_size_classnet = images_classnet.shape.as_list()[1]
    if self.glimpse_shape_classnet[0] < 0:
      self.glimpse_shape_classnet = tuple([int(
          image_size_classnet / image_size_saccader *
          self.glimpse_shape[0])] * 2)
    self.glimpse_shape = self.glimpse_shape_classnet

    images_glimpse_t = []
    for locations in locations_t:
      images_glimpse = utils.extract_glimpse(
          images_classnet, size=self.glimpse_shape_classnet, offsets=locations)
      images_glimpse_t.append(images_glimpse)

    batch_size = tf.shape(images_classnet)[0]
    images_glimpse_t = tf.concat(images_glimpse_t, axis=0)

    variables_before = set(tf.global_variables())
    reuse = True if self.var_list_classnet else False
    with tf.variable_scope(self.variable_scope_classnet, reuse=reuse):
      if self.classnet_type == "nasnet":
        classnet_config = nasnet.large_imagenet_config()
        classnet_config.use_aux_head = 0
        classnet_config.drop_path_keep_prob = 1.0
        with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
          classnet_logits, endpoints_ = nasnet.build_nasnet_large(
              images_glimpse_t, self.num_classes,
              is_training=is_training_classnet,
              config=classnet_config)
      elif self.classnet_type == "resnet_v2_50":
        network = nets_factory.get_network_fn(
            "resnet_v2_50", self.num_classes, is_training=is_training_classnet)
        classnet_logits, endpoints_ = network(images_glimpse_t)

    endpoints["classnet"] = endpoints_
    variables_after = set(tf.global_variables())
    logits_t = tf.reshape(classnet_logits, (num_times, batch_size, -1))
    logits = tf.reduce_mean(logits_t, axis=0)
    if not reuse:
      self.var_list_saccader = self.var_list_classification + self.var_list_location
      self.var_list_classnet = [
          v for v in list(variables_after-variables_before)
          if "global_step" not in v.op.name]
      self.var_list.extend(self.var_list_classnet)
      self.init_op = tf.variables_initializer(var_list=self.var_list)

    return logits, locations_t, best_locations_t, endpoints
