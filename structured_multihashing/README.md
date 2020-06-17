# Structured Multihashing for Model Compression

## Overview
Structured Multi-Hashing (SMH) is an approach for compressing the size of deep
neural networks. Despite the success of deep networks, state-of-the-art models
are often too large to be deployed on low resource devices or common server
configurations in which multiple models are held in the memory.  

Unlike some existing model compression methods, such as model quantization and
model pruning, SMH reduces the number of learnable variables of a model  by
mapping model weights into a variable pool with a smaller size, which is shared
among all the layers. Traditional weight hashing techniques have hash collisions
between model weights, and lack memory locality which makes them slow to train
and deploy. SMH addresses both of these problems by defining a structured
mapping which is implemented as an efficient matrix-matrix multiplication.  

When applied to several popular state-of-the-art model families, e.g. ResNet,
MobileNet and EfficientNet, SMH can successfuly reduce model size without loss
in quality. Most notably on EfficentNet-architectures where it has 40% size
reduction and 2% accuracy gains. More details and results can be found in our
CVPR 2020 paper ["Structured Multi-Hashing for Model
Compression"](https://arxiv.org/pdf/1911.11177.pdf).

## How to Use
To illustrate how to apply SMH, we use the following toy neural network which
contains one 2D convolution layer, a 2D max pooling layer, and a fully connected
layer. We show both a TF 2.x and TF 1.x implementation of the model, which will
be used as an example in the rest of the tutorial.

```python
# TF2.x implementation with Keras.
import tensorflow as tf

def toy_network_tf2(images):
  net = tf.compat.v2.keras.layers.Conv2D(
      16, (5, 5), padding='same', activation='relu')(images)
  net = tf.compat.v2.keras.layers.MaxPool2D(
      pool_size=(2, 2), strides=2)(net)
  net = tf.compat.v2.keras.layers.Flatten()(net)
  logits = tf.compat.v2.keras.layers.Dense(10)(net)

  return logits
```

```python
# TF1.x implementation with tf_slim.
import tensorflow as tf
import tf_slim as slim

def toy_network_tf1(images, scope='ToyNetwork'):
  with tf.compat.v1.variable_scope(scope, 'ToyNetwork'):
    net = slim.conv2d(images, 16, [5, 5], scope='conv')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool')
    net = slim.flatten(net)
    logits = slim.fully_connected(net, 10, scope='fully_connected')

  return logits
```

### Step 1: Determine the variable pool size.
The size of the variable pool represents an upper bound of the number of
learnable variables in the original model before compression with SMH. It may
not be exactly the same with the number of learnable weights in the model, but
it must be larger than that. For the example network, suppose the input image
tensors have dimension of [batch_size, 128, 128, 3]. In this case, we could
choose variable pool size to be 660,000. The following table explains how the
pool size is determined.

| layer | number of learnable variables |
| ----- | ----------------------------- |
| conv2d | 5 * 5 * 3 * 16 + 16 = 1216 |
| fully_connected | 64 * 64 * 16 * 10 + 10 = 655370 |
|------- | ---------------------------- |
| total | 656586 < 660000 |

### Step 2: Create the variable pool and variable factory.
Given the variable pool size and compression ratio, the variable pool and
variable factory, which allocate variables from the pool to models, can be
created with the following snippets:

```python
from smh import variable_factory
from smh import variable_pool

pool = variable_pool.ProductVariablePool(
    trainable=True,
    pool_size=7e5,
    fraction=0.4  # Create a model that is 40% of the original size.
    )
factory = variable_factory.VariableFactory(
    variable_pool = pool,
    apply_to = '.*',  # Apply to every variable created in the model.
    modifier = variable_factory.fanout_scale_modifier)
```

### Step 3: Apply to the existing code that creates TensorFlow models.
Depending on whether the model is implemented with TensorFlow 2.x or TensorFlow
1.x, the way to apply SMH compression is slightly different, which will be
illustrated separately below.

#### Model implemented with TensorFlow 2.x and Keras
With TensorFlow 2.x models, SHM compression can be applied via
variable_creator_scope as shown in the code snippets below.

```python
with tf.variable_creator_scope(factory.custom_getter_tf2):
  ...
  logits = toy_network_tf2(images)
  ...
```

However, variable_scope has been removed in TensorFlow 2.x. As a result,
variable names cannot be used to identify which layer they belong to. Thus, the
regular expression passed to `apply_to` argument can only be used to choose to
compress kernel variables or bias variables of all the layers in the model. For
exmaple, the code snippets below will only apply SMH compression to the kernel
variables.

```python
kernel_factory = variable_factory.VariableFactory(
    variable_pool = pool,
    apply_to = '.*kernel.*',  # Kernel variables have default name of 'kernel' for Keras layers.
    modifier = variable_factory.fanout_scale_modifier)
# This will apply SMH compression to the kernel variables of all the layers in
# the Keras model created by toy_network_tf2 method.
with tf.variable_creator_scope(kernel_factory.custom_getter_tf2):
  ...
  logits = toy_network_tf2(images)
  ...
```

To apply SMH compression to certain layers of TensorFlow 2.x models,
the selected layers have to be manually wrapped by the variable_creator_scope in
the model builder function. The following example will only apply SMH
compression to the `Conv2D` layer of the toy Keras model.

```python
def toy_network_tf2(images, var_factory):
  with tf.variable_creator_scope(var_factory.custom_getter_tf2):
    net = tf.compat.v2.keras.layers.Conv2D(
        16, (5, 5), padding='same', activation='relu')(images)
  net = tf.compat.v2.keras.layers.MaxPool2D(
      pool_size=(2, 2), strides=2)(net)
  net = tf.compat.v2.keras.layers.Flatten()(net)
  logits = tf.compat.v2.keras.layers.Dense(10)(net)

  return logits


logits = toy_network_tf2(images, factory)
```

#### Model implemented with TensorFlow 1.x
With TensorFlow 1.x model, the SMH compression can be applied via the variable
custom getter argument of tf.variable_scope.

```python
with tf.compat.v1.variable_scope('SMH', custom_getter=factory.custom_getter):
  ...
  logits = toy_network_tf1(images)
  ...
```

Since in TensorFlow 1.x, variable_scope can be used to append prefixes to
variable names, appropriate regular expressions can be passed to `apply_to`
argument of `VariableFactory` to only compress variables of certain layers whose
variable names match the given regular expression. For example,

```python
conv_factory = variable_factory.VariableFactory(
    variable_pool = pool,
    apply_to = '.*conv.*',  # Apply to variables whose name contains 'conv'.
    modifier = variable_factory.fanout_scale_modifier)
with tf.compat.v1.variable_scope(
    'SMH', custom_getter=conv_factory.custom_getter):
  ...
  logits = toy_network_tf1(images)
  ...
```

In this case, the SMH will only compress the variables of the conv2d layer in
the example toy network since it has a scope name of 'conv' which will make the
names of all its variables match the given regular expression '*conv*'.

### VariableFactory Modifier
A VariableFactory modifier is a function passed to the `modifier` argument of
the constructor of the `VariableFactory` class. The modifier function only takes
a single instance of `tf.Tensor` as its arguments, and it will be applied to the
variable tensor for additional modifications before returned by
`VariableFactory.custom_getter` or `VariableFactory.custom_getter_tf2`. Users
are encouraged to define their own modifier functions when necessary. These
modifier functions are provided in the library:

* **relu_modifier**: Apply `Relu` function to the input tensor.  
* **constant_fanout_scale_modifier**: Apply a fanout scale used in EfficientNet
  to the input tensor.
* **variable_scale_modifier**: Multiply a learnable tensor to the input tensor.

## Contact

### Maintainers
* Elad Eban (elade@google.com)
* Yair Movshovitz-Attias (yairmov@google.com)

### Contributors
* Hao Wu (haou@google.com)
