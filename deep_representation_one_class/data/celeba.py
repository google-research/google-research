# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""CelebA data."""

import os
import numpy as np
from deep_representation_one_class.data.cifar import CIFAR


class CelebA(CIFAR):
  """CelebA data loader for anomaly detection."""

  def __init__(self,
               root,
               dataset='celeba',
               category='Eyeglasses',
               input_shape=(64, 64, 3)):
    assert input_shape[0] in [64, 224]
    self.attr_keys = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    self.np_path = os.path.join(root, 'celeba_npy')
    self.dataset = dataset
    self.category = category
    self.input_shape = input_shape
    self.db_train, self.db_test = self.prepare_numpy_data(size=input_shape[0])

    attributes = [category]
    attr_indices = np.array([
        i for i in range(len(self.attr_keys)) if self.attr_keys[i] in attributes
    ])
    # train set
    attr_of_interest = self.db_train[1][:, attr_indices]
    normal_indices = np.sum(attr_of_interest, axis=1) == 0
    x_train = self.db_train[0][normal_indices]
    y_train = np.sum(self.db_train[1][normal_indices][:, attr_indices], axis=1)
    # test set
    attr_of_interest = self.db_test[1][:, attr_indices]
    y_test = np.asarray(np.sum(attr_of_interest, axis=1) > 0, np.float32)
    x_test = self.db_test[0]
    self.trainval_data = [
        x_train, y_train,
        np.expand_dims(np.arange(len(y_train)), axis=1)
    ]
    self.test_data = [
        x_test, y_test,
        np.expand_dims(np.arange(len(y_test)), axis=1)
    ]
