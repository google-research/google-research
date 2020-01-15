# Ops for Gradient Based Pruning

This document describes the Tensorflow ops that support first-order and
second-order gradient based pruning. This is an extension of the existing
[model pruning library](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/model_pruning/README.md)
in Tensorflow. The motivation of this approach is described in
[gradient_based_pruning.pdf](https://github.com/google-research/google-research/tree/master/model_pruning/gradient_based_pruning.pdf).

## Model Creation

The model creation API remains the same as the original model pruning library.
Mask, threshold, and gradient variables are all wrapped under the `apply_mask`
function provided in pruning.py.

For example:

```python
conv = tf.nn.conv2d(images, pruning.apply_mask(weights), stride, padding)
```

This creates a convolution layer with additional mask, threshold, gradient, and
decay variables. If the `prune_option` is set to gradient based, gradient
variable will be updated when `apply_mask` function is called.

## Gradient-related Hyperparameters

We added two new hyperparameters on top of the original hyperparameters in
model pruning library.

|Hyperparameter               | Type    | Default       | Description |
|:----------------------------|:-------:|:-------------:|:------------|
|prune_option|string|'weight'|Pruning option. Other options are 'first_order_gradient' and 'second_order_gradient'|
|gradient_decay_rate|float|0.99|That is used to control the decay strength when calculating moving average for gradients.|

*Note: gradient based pruning doesn't support block sparsity*

## Adding Pruning Ops to the Training Graph

Adding pruning ops to the training graph, which monitors the distribution of
each layer's weight, gradient, and mask, also remains the same as the original
model pruning library. An example is as follows.

```python
tf.app.flags.DEFINE_string(
    'pruning_hparams', '',
    """Comma separated list of pruning-related hyperparameters""")

with tf.graph.as_default():

  # Create global step variable
  global_step = tf.train.get_or_create_global_step()

  # Parse pruning hyperparameters
  pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)

  # Create a pruning object using the pruning specification
  p = pruning.Pruning(pruning_hparams, global_step=global_step)

  # Add conditional mask update op. Executing this op will update all
  # the masks in the graph if the current global step is in the range
  # [begin_pruning_step, end_pruning_step] as specified by the pruning spec
  mask_update_op = p.conditional_mask_update_op()

  # Add summaries to keep track of the sparsity in different layers during training
  p.add_pruning_summaries()

  with tf.train.MonitoredTrainingSession(...) as mon_sess:
    # Run the usual training op in the tf session
    mon_sess.run(train_op)

    # Update the masks by running the mask_update_op
    mon_sess.run(mask_update_op)

```
Ensure that `global_step` is being [incremented](https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#minimize), otherwise pruning will not work!

## Removing Pruning Ops from Trained Graph

Once the model is trained, it is necessary to remove the auxiliary variables
(mask, threshold, gradients) and pruning ops added to the graph in the steps
above. This can be accomplished using the `strip_pruning_vars` utility. This
utility is also extended to support gradient variables that are newly added
in this library.

The usage is the same as the original model pruning library.

```shell
$ bazel build -c opt model_pruning:strip_pruning_vars
$ bazel-bin/model_pruning/strip_pruning_vars
--checkpoint_dir=/path/to/checkpoints/
--output_node_names=graph_node1,graph_node2 --output_dir=/tmp
--filename=pruning_stripped.pb
```

## References

Michael Zhu and Suyog Gupta, “To prune, or not to prune: exploring the efficacy
of pruning for model compression”, *2017 NIPS Workshop on Machine Learning of
Phones and other Consumer Devices*.

Yann Le Cun, John S. Denker, and Sara A. Solla, "Optimal brain damage",
*Advances in neural information processing systems 2*, Morgan Kaufmann
Publishers Inc., San Francisco, CA, USA 598-605.

## Authors
Yang Yang (yyn@google.com) and Rina Panigrahy (rinap@google.com)

## Acknowledgement
We would like to thank Suyog Gupta and Badih Ghazi for useful discussions.

## Note
This is not an officially supported Google product.
