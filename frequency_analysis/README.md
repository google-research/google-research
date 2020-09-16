## Generate Fourier heat maps of image classification models

Link to paper: https://arxiv.org/pdf/1906.08988.pdf

The goal of this study is to analyze how deep learning models, in particular,
image classification models respond to perturbations with different frequency
patterns. We generate the Fourier heat map of models to visualize how the
outputs of the layers and test error change when we add different frequency
basis vectors to the input images.

Usage: First, define a `TensorFlowNeuralNetwork` object. This class is defined
in `freq_heatmap.py`, and can be used as an interface between the frequency
analysis and the TensorFlow sessions.

```python
class TensorFlowNeuralNetwork(object):
  """The interface between TensorFlow and the heat map generator."""

  def __init__(self, sess, image_ph, label_ph, tensor_list, eval_tensor):
    """Initializing TensorFlowNeuralNetwork.

    Args:
      sess: a tensorflow session.
      image_ph: a tensorflow placeholder for input images. This tensor should
        have shape [num_examples, height, width] or
        [num_examples, height, width, num_channels].
      label_ph: a tensorflow placeholder for the labels of images. Must be shape
        [num_examples].
      tensor_list: a list of tensors to evaluate. The tensors in this list can
        have arbitrary and different shapes. These tensors correspond to the
        outputs of the layers that we are interested in. To conduct frequency
        analysis, first feed clean images and evaluate the tensors, then feed
        corrupted images using Fourier basis and evaluate the tensors, and
        finally compare the difference between the two evaluations.
      eval_tensor: a tensor of shape [], i.e., a scalar, corresponding to an
        evaluation criterion on a batch. This can be the test accuracy, or any
        loss (such as cross-entropy loss).
    """
```

Then, call `generate_freq_heatmap` in `freq_heatmap.py` to generate the Fourier
heat maps.

```python
def generate_freq_heatmap(neural_network,
                          images,
                          labels,
                          custom_basis=None,
                          perturb_norm=1.0,
                          batch_size=-1,
                          clip_min=None,
                          clip_max=None,
                          rand_flip=False,
                          seed=None,
                          relative_scale=True):
  """Generate frequency heat map.

  We conduct the frequcy analysis in the following way: 1) feed the images to
  the network and record the values of tensors in tensor_list, and 2) feed the
  images with Fourier-basis perturbations, and record the values of tensors in
  tensor_list, and 3) compare the difference between the tensors.

  Args:
    neural_network: a TensorFlowNeuralNetwork object.
    images: a numpy array of images with shape [num_examples, height, width] or
      [num_examples, height, width, num_channels]. The shape of images should
      be the same as the image placeholder in neural_network.
    labels: a numpy array for the labels of images with shape [num_examples].
    custom_basis: a numpy array of shape [height, width, height, width], with
      each slice [i, j, :, :] being the (i, j) basis vector. If None, we first
      generate the basis.
    perturb_norm: the l_2 norm of the Fourier-basis perturbations.
    batch_size: the batch size when computing the frequency heatmap. If the
      number of examples in image_np is large, we may need to compute the
      heatmap batch by batch. If batch_size is -1, we use the entire image_np.
    clip_min: lower bound for clipping operation after adding perturbation. If
      None, no lower clipping.
    clip_max: upper bound for clipping operation after adding perturbation. If
      None, no upper clipping.
    rand_flip: whether or not to randomly flip the sign of basis vectors.
    seed: numpy random seed for random flips.
    relative_scale: whether or not to return relative scale of the tensor change
      across all the frequency components. If True, the maximum change is
      normalized to be 1; otherwise, return the actual value of the model change
      under Fourier-basis perturbation, averaged across the input images.

  Returns:
    heatmap_list: a list of numpy arrays, each has shape [height, width]. The
      heatmaps in the list correspond to the tensors in the `tensor_list` in
      `neural_network`.
    eval_heatmap: a numpy array of shape [height, width], each entry in the
      array corresponds to the evaluation criterion (`eval_tensor` in
      `neural_network`) under the Fourier basis perturbation, averaged across
      the batches.
    clean_eval: a scalar corresponding to the evaluation criterion on image_np,
      averaged across the batches.
  """
```
