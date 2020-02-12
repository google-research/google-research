# The Tree Ensemble Layer

This repository contains a new *tree ensemble layer* (TEL) for neural networks. The layer is differentiable so SGD can be used to train the neural network (including TEL). The layer supports conditional computation for both training and inference, i.e., when updating/evaluating a certain node in the tree, only the samples that reach that node are used in computations (this is to be contrasted with the dense computations in neural networks). We provide a low-level Tensorflow implementation along with a high-level Keras API.

More details to be added soon.

## Installation
The installation instructions below assume that you have Python, TensorFlow, and GCC already installed.

From inside the directory containing TEL's code, run the following commands:
```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared neural_trees_ops.cc neural_trees_kernels.cc neural_trees_helpers.cc -o neural_trees_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```
Note: On OS X, add the flag "-undefined dynamic_lookup" (without quotes) to the last command above.

## Example Usage
In Keras, the layer can be used as follows:
```python
import tensorflow as tf
from tensorflow import keras

# Import the layer.
from tel import TEL
# The documentation of TEL can be accessed as follows
?TEL

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
model.compile(loss='mse',  optimizer='adam', metrics=['mse'])
result = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
