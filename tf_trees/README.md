# The Tree Ensemble Layer

This repository contains a new *tree ensemble layer* (TEL) for neural networks. The layer is differentiable so SGD can be used to train the neural network (including TEL). The layer supports conditional computation for both training and inference, i.e., when updating/evaluating a certain node in the tree, only the samples that reach that node are used in computations (this is to be contrasted with the dense computations in neural networks). We provide a low-level TensorFlow implementation along with a high-level Keras API.

More details are available in our paper *The Tree Ensemble Layer: Differentiability meets Conditional Computation* ([arXiv link](https://arxiv.org/abs/2002.07772v1)).

## Installation
The installation instructions below assume that you have Python and TensorFlow already installed. Use Method 1 if you have installed TensorFlow from source. Otherwise, use Method 2.

### Method 1: Compile using Bazel
If you have TensorFlow sources installed, you can make use of TensorFlow's build system to compile TEL.
First, copy the file "BUILD" (available in the tf_trees directory) to the directory "tensorflow/core/user_ops".
Then, from inside the tf_trees directory, run the following command:
```bash
bazel build --config opt //tensorflow/core/user_ops:neural_trees_ops.so
```

### Method 2: Compile using G++
Use this method if you have installed TensorFlow from binary. From inside the tf_trees directory, run the following commands:
```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared neural_trees_ops.cc neural_trees_kernels.cc neural_trees_helpers.cc -o neural_trees_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```
If the installation is unsuccessful due to the compiler being too old to be supported by Eigen, use `g++ -std=c++17` instead for installation:
```bash
g++ -std=c++17 -shared neural_trees_ops.cc neural_trees_kernels.cc neural_trees_helpers.cc -o neural_trees_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

Note: On OS X, add the flag "-undefined dynamic_lookup" (without quotes) to the last command above.

## Example Usage
In Keras, the layer can be used as follows:
```python
from tensorflow import keras
# Make sure the tf_trees directory is in the search path.
from tf_trees import TEL

# Define the tree layer: here we choose 10 trees, each of depth 3.
# output_logits_dim is the dimension of the tree output.
tree_layer = TEL(output_logits_dim=2, trees_num=10, depth=3)

# tree_layer can be used as part of a Keras sequential model.
model = keras.Sequential()
# ... Add your favorite layers here ...
model.add(keras.layers.BatchNormalization())
model.add(tree_layer)
# ... Add your favorite layers here ...

# See demo.py for a full example.
```

## Citing TEL
If you find this work useful in your research, please consider citing the following paper:
```
@article{hazimeh2020tree,
    title={The Tree Ensemble Layer: Differentiability meets Conditional Computation},
    author={Hussein Hazimeh and Natalia Ponomareva and Petros Mol and Zhenyu Tan and Rahul Mazumder},
    year={2020},
    eprint={2002.07772},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
