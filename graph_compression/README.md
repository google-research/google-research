# Matrix Compression Library

This document describes an experimental API that facilitates matrix compression
of a neural network's weight tensors. The API helps inject the necessary
tensorflow operations into the training graph so the model can be compressed
while it is being trained.

Full documentation can be found
[here](https://drive.google.com/file/d/1843aNpKx_rznpuh9AmEshgAKmISVdpJY/view).

## Table of contents

1.  [Library Overview](#library-overview)
2.  [Model creation](#model-creation)
3.  [Hyperparameters for compression](#hyperparameters)
    -   [Smoothed compression](#smoothed-compression)
4.  [Adding compression ops to the training graph](#adding-compression-ops)
5.  [Example](#example)

### Library overview <a name="library-overview"></a>

1.  **MatrixCompressorInterface** - used to implement any matrix compression
    algorithm in the method
2.  **CompressionOpInterface** - used to create a tensorflow operator-like
    object that injects any matrix compression method dynamically into a
    tensorflow layer.
3.  **ApplyCompression** - convenience wrapper class that can be used directly
    or extended for novel compression operator types; used to repeatedly invoke
    the compression operator to different layers in a model.
4.  **CompressionWrapper** - wrapper module used to create the proper
    ApplyCompression implementation for the compression_option (method) of
    choice.

### Model creation <a name="model-creation"></a>

The first step involves creating an ApplyCompression object, with the desired
compression parameters. This object then is used to compress the model weights
and use these compressed weights during the forward execution of the graph.
Matrices are compressed to the rank specified in the compression parameters,
provided at the start. To apply the compression, the weight tensor of the layer
should be wrapped with the compression object's 'apply_compression' method,
provided in
[compression_op.py](https://github.com/google-research/google-research/tree/master/graph_compression/compression_lib/compression_op.py).For
an example, see the [section below](#adding-compression-ops).

### Hyperparameters for compression <a name="hyperparameters"></a>

The pruning library allows for specification of the following hyper parameters:

Hyperparameter         | Type    | Default           | Description
:--------------------- | :-----: | :---------------: | :----------
name                   | string  | model_compression | Name of the compression specification. Used for adding summaries and ops under a common tensorflow name_scope.
alpha_decrement_value  | float   | 0.01              | Real number by which alpha is decremented at each update.
begin_compression_step | integer | 0                 | Global step at which to begin compression.
end_compression_step   | integer | -1                | Global step at which to terminate compression. Defaults to -1 implying compression continues till the training stops.
compression_frequency  | integer | 10                | Intervals at which compression is applied and compression parameters updated.
compression_option     | integer | 0                 | Indicates what type of factorization/compression to use (see the list below for the algorithm options).
rank                   | integer | 100               | Factorization rank (r), where if A = BC. See definition below of how rank (r) is used to compute final weights matrix dimensions.
update_option          | integer | 0                 | Indicates how update logic is being run: 0 - use tensorflow operations for updates; 1 - use python functions for updates.
use_tpu                | boolean | False             | **Experimental flag** - training using TPUs

#### Compression Methods & Algorithms (compression_option param)

1.  Low Rank Approximation
2.  Simhash

#### Decomposed Matrix Dimensions

The hyperparameter rank (r) is used to compute the new ranks as such: (rank of
A) * (100 / r) + 1. For simhash compression, the value r provided should be the
ratio value you would like divided by 8 (i.e. 300 / 8 -> same as using r = 300
in the equation above). This is because simhash compression represents values as
bits (rather than bytes) therefore the true rank is the size of the array
divided by 8.

#### Computing Compression Ratio

If the original weights were m-by-n and the compressed decomposition B\*C is
(m-by-k)\*(k-by-n), then the compression ratio is (m\*k + k\*n) / (m\*n).

#### Smoothed compression <a name="smoothed-compression"></a>

A gradually increasing alpha value is used to smooth the compression from
start_step to end_step. This way the model gradually moves from the full weights
matrix to a compressed one. For example, in the low-rank approximation scheme,
the weight matrix that is used in the training process is W = (alpha) * A + (1 -
alpha) * BC. This alpha value is decremented over time from alpha = 1 to alpha =
0, using the alpha_decrement_value at intervals of compression_frequency.

### Adding compression ops to the training graph <a name="adding-compression-op"></a>

```python
# Parse compression hyperparameters
compression_hparams = compression.CompressionOp.get_default_hparams().parse(
      hparams)

# Create a compression object using the compression hyperparameters
compression_obj = compression_wrapper.get_apply_compression(
    compression_hparams, global_step=global_step)

# somewhere in the model, compute the compressed weights
local = tf.nn.relu(
         tf.matmul(reshape, compression_obj.apply_compression(weights, scope)) +
         biases,
         name=scope.name)

all_update_op = [apply_gradient_op, ...] # all existing model updates
# Run compression update steps with all the other updates. Example below is
# assuming update_option=0.
all_update_op.append(compression_obj.all_update_op())

with tf.control_dependencies(all_update_op):
  train_op = tf.no_op(name='train')
```

Ensure that `global_step` is being incremented, otherwise compression will not
work!

#### Example Usage <a name="example"></a>

As an example, the cifar10 model provided in Tensorflow’s
[Advanced Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/deep_cnn)
(see page for more details) has been modified to incorporate the compression
library:

*   [cifar10_compression.py](https://github.com/google-research/google-research/tree/master/graph_compression/compression_lib/examples/cifar10/cifar10_compression.py)
    creates the deep CNN and adds the weight compression to the fully-connected
    layers.
*   [cifar10_train.py](https://github.com/google-research/google-research/tree/master/graph_compression/compression_lib/examples/cifar10/cifar10_train.py)
    creates the compression object and provides it to the training graph
    (described above) to use.

To train the compression version of cifar10 (make sure you're working in a
properly configured virtualenv - as setup using the
[run.sh](https://github.com/google-research/google-research/tree/master/graph_compression/run.sh)
script):

```shell
$ python cifar10_train.py --compression_hparams=name=cifar10_compression,alpha_decrement_value=0.005,begin_compression_step=40000,end_compression_step=100000,compression_frequency=100,compression_option=1,use_tpu=True,update_option=0,rank=200 --max_steps 120000
```

Eval:

```shell
$ python cifar10_eval.py --compression_hparams=name=cifar10_compression,alpha_decrement_value=0.005,begin_compression_step=40000,end_compression_step=100000,compression_frequency=100,compression_option=1,use_tpu=True,update_option=0,rank=200 --run_once
```

Authors: Rina Panigrahy (corresponding author -- email: rinap@google.com),
Lucine Oganesian, Sudeshna Roy, with support from: Badih Ghazi for helpful
contributions, Rasmus Pagh (pagh@google.com,
[doc](https://drive.google.com/file/d/10TWVnHExdWdQ8DyPELV18Rq92zutSzp9/view?usp=sharing))
for the simhash code, Zoya Svitkina for code reviews, and Suyog Gupta for
consultations.
