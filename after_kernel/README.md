## Running an experiment
This code runs an experiment with the "after kernel".  The after kernel is
like the neural tangent kernel, but after training.  For example, to compare
the translation invariance of the after kernel to the neural tangent kernel after
training for 10 epochs on a binary classification problem with MNIST classes
3 and 8, run

```
python -m run_ak_experiment --dataset=mnist \
                            --plus_class=8 \
                            --minus_class=3 \
                            --num_translations=1000 \
                            --num_epochs=10
```

## Command-line flags
Here is a list of all of the command-line flags.  All of them have default 
values.
- num_epochs (integer): number of epochs of training of the neural network
- test_accuracy_output (string): name of a file for output of test accuracy
and other statistics
- num_translations (integer): the number of test examples to use to test the
translation-invariance of the after kernel
- num_zooms (integer): the number of test examples to use to test the
zoom-invariance of the after kernel
- num_swaps (integer): the number of test examples to use to test the
invariance of the after kernel to swaps of the upper left and lower right
quadrants of the input image
- num_rotations (integer): the number of test examples to use to test the
invariance of the kernel to rotations
- ker_alignment_sample_size (integer): the number of test examples to use
to test the alignment between the after kernel and its conjugate projection
- effective_rank_sample_size (integer): the number of test examples to use
to evaluate the effective rank of the Gram matrix
- rotation_angle (float): the angle, in radians to use when measuring the
rotation invariance
- use_augmentation (boolean): whether or not to use data augmentation
when training the neural network used to generate the after kernel
- train_linear_model (boolean): whether or not to train a linear model using
the after-kernel embeddings, and evaluate its accuracy
- use_conjugate_kernel (boolean): whether or not to replace the after kernel
with its conjugate projection
- plus_class (integer): the class to use for positive examples, when extracting
a two-class classification problem
- minus_class (integer): the class to use for negative examples, when extracting
a two-class classification problem
- dataset (string, choose one of "mnist" or "cifar10"): which dataset to
use
- num_runs (integer): the number of times to repeat the experiment, followed
by averaging the results
- nn_architecture (strong, choose one of "VGG", "MLP", or "Sum"): which
architecture to use
- model_summary_output (string): name of a file for output of a summary of
the neural network model used to generate the after kernel
- filter_size (integer): size of the filters, when a convolutional network
is used
- pool_size (integer): size of the regions to pool, when a convolutional network
is used
- num_parameters (integer): a rough guide for the size of the neural network
model used to define the after kernel
- num_blocks (integer): the number of blocks (consisting of convolutional
layers followed by max-pooling) in the network, if a VGG-like network is used
- num_layers (integer): the number of layers in the neural network used to
generate the after kernel

## Note
This code depends on `tensorflow-addons`.  As of this writing, 
`tensorflow-addons` is incompatible with the most recent version of python.  
I worked around this by using `pyenv` to install an earlier version of python 
before installing `tensorflow-addons`.
