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

"""Network Architectures."""

import dataclasses
from typing import Callable, Optional, Text, Tuple

import tensorflow as tf


@dataclasses.dataclass
class NetworkArchitecture:
  """Class to describe network architecture."""
  name: Text
  base: Callable  # tf.keras.application.model pylint: disable=g-bare-generic
  preprocessing: Callable  # tf.keras.application.preprocessing pylint: disable=g-bare-generic
  target_shape: Tuple[int, int] = (224, 224)
  weights: Optional[Text] = None

  def set_weights(self, source_dataset_name):
    if source_dataset_name == 'random':
      self.weights = None
    else:
      self.weights = source_dataset_name

  def get_target_model(self,
                       num_classes = 0,
                       base_trainable = False):
    """Get target prediction model."""
    base_model = self.base(
        include_top=False,
        weights=self.weights,
        input_tensor=tf.keras.Input(shape=(
            self.target_shape[0], self.target_shape[1], 3)))

    base_model.trainable = base_trainable

    target_model_layers = [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(name='avg_pool'),
    ]

    if num_classes > 0:
      target_model_layers.append(
          tf.keras.layers.Dense(num_classes),  # Activation in softmax loss.
          )

    return tf.keras.Sequential(target_model_layers)

  def get_feature_model(self):
    return self.get_target_model(base_trainable=False)

  def get_prediction_model(self):
    return self.base(
        include_top=True,
        weights=self.weights,
        input_tensor=tf.keras.Input(shape=(
            self.target_shape[0], self.target_shape[1], 3)))


NETWORK_ARCHITECTURES = [
    NetworkArchitecture('ResNet50', tf.keras.applications.resnet.ResNet50,
                        tf.keras.applications.resnet.preprocess_input),
    NetworkArchitecture('ResNet101', tf.keras.applications.resnet.ResNet101,
                        tf.keras.applications.resnet.preprocess_input),
    NetworkArchitecture('ResNet152', tf.keras.applications.resnet.ResNet152,
                        tf.keras.applications.resnet.preprocess_input),
    NetworkArchitecture('ResNet50v2',
                        tf.keras.applications.resnet_v2.ResNet50V2,
                        tf.keras.applications.resnet_v2.preprocess_input),
    NetworkArchitecture('ResNet101v2',
                        tf.keras.applications.resnet_v2.ResNet101V2,
                        tf.keras.applications.resnet_v2.preprocess_input),
    NetworkArchitecture('ResNet152v2',
                        tf.keras.applications.resnet_v2.ResNet152V2,
                        tf.keras.applications.resnet_v2.preprocess_input),
    NetworkArchitecture('VGG16', tf.keras.applications.vgg16.VGG16,
                        tf.keras.applications.vgg16.preprocess_input),
    NetworkArchitecture('VGG19', tf.keras.applications.vgg19.VGG19,
                        tf.keras.applications.vgg19.preprocess_input),
    NetworkArchitecture('DenseNet121',
                        tf.keras.applications.densenet.DenseNet121,
                        tf.keras.applications.densenet.preprocess_input),
    NetworkArchitecture('DenseNet169',
                        tf.keras.applications.densenet.DenseNet169,
                        tf.keras.applications.densenet.preprocess_input),
    NetworkArchitecture('DenseNet201',
                        tf.keras.applications.densenet.DenseNet201,
                        tf.keras.applications.densenet.preprocess_input),
    NetworkArchitecture('XCeption', tf.keras.applications.xception.Xception,
                        tf.keras.applications.xception.preprocess_input),
    NetworkArchitecture('MobileNet', tf.keras.applications.mobilenet.MobileNet,
                        tf.keras.applications.mobilenet.preprocess_input),
    NetworkArchitecture('MobileNetV2',
                        tf.keras.applications.mobilenet_v2.MobileNetV2,
                        tf.keras.applications.mobilenet_v2.preprocess_input),
    NetworkArchitecture('MobileNetV3', tf.keras.applications.MobileNetV3Large,
                        tf.keras.applications.mobilenet_v3.preprocess_input),
    NetworkArchitecture('NasNetMobile',
                        tf.keras.applications.nasnet.NASNetMobile,
                        tf.keras.applications.nasnet.preprocess_input),
    NetworkArchitecture('NasNetLarge',
                        tf.keras.applications.nasnet.NASNetLarge,
                        tf.keras.applications.nasnet.preprocess_input),
    NetworkArchitecture('EfficientNetB0',
                        tf.keras.applications.efficientnet.EfficientNetB0,
                        tf.keras.applications.efficientnet.preprocess_input),
    NetworkArchitecture('EfficientNetB1',
                        tf.keras.applications.efficientnet.EfficientNetB1,
                        tf.keras.applications.efficientnet.preprocess_input),
    NetworkArchitecture('EfficientNetB2',
                        tf.keras.applications.efficientnet.EfficientNetB2,
                        tf.keras.applications.efficientnet.preprocess_input),
    NetworkArchitecture('EfficientNetB3',
                        tf.keras.applications.efficientnet.EfficientNetB3,
                        tf.keras.applications.efficientnet.preprocess_input),
    NetworkArchitecture('EfficientNetB4',
                        tf.keras.applications.efficientnet.EfficientNetB4,
                        tf.keras.applications.efficientnet.preprocess_input),
    NetworkArchitecture('EfficientNetB5',
                        tf.keras.applications.efficientnet.EfficientNetB5,
                        tf.keras.applications.efficientnet.preprocess_input),
    NetworkArchitecture('EfficientNetB6',
                        tf.keras.applications.efficientnet.EfficientNetB6,
                        tf.keras.applications.efficientnet.preprocess_input),
    NetworkArchitecture('EfficientNetB7',
                        tf.keras.applications.efficientnet.EfficientNetB7,
                        tf.keras.applications.efficientnet.preprocess_input),

]

NETWORK_ARCHITECTURES = {
    na.name.lower(): na for na in NETWORK_ARCHITECTURES
}
