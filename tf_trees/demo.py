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

from tensorflow import keras
# Make sure the tf_trees directory is in the search path.
from tf_trees import TEL

# The documentation of TEL can be accessed as follows
print(TEL.__doc__)

# We will fit TEL on the Boston Housing regression dataset.
# First, load the dataset.
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Define the tree layer; here we choose 10 trees, each of depth 3.
# Note output_logits_dim is the dimension of the tree output.
# output_logits_dim = 1 in this case, but should be equal to the
# number of classes if used as an output layer in a classification task.
tree_layer = TEL(output_logits_dim=1, trees_num=10, depth=3)

# Construct a sequential model with batch normalization and TEL.
model = keras.Sequential()
model.add(keras.layers.BatchNormalization())
model.add(tree_layer)

# Fit a model with mse loss.
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
result = model.fit(
    x_train, y_train, epochs=100, validation_data=(x_test, y_test))
