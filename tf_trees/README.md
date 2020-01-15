# The Tree Ensemble Layer

This repository contains a new *tree ensemble layer* which can be used anywhere in a neural network. We provide a low-level Tensorflow implementation along with a high-level Keras API.

The layer is differentiable so SGD can be used to train the neural network (including the tree layer). The layer supports conditional computation for training and inference: when updating/evaluating a certain node in the tree, only the samples that reach that node are used in computations (this is to be contrasted with the dense computations in neural networks).

## Example Usage
In Keras, the layer can be used as follows:
```python
from keras.models import Sequential
from neural_trees_layer import NeuralTrees

model = Sequential()
# Add you favorite layers here...
tree_layer = NeuralTrees(output_logits_dim=10, trees_num=5, depth=2)
model.add(tree_layer)
# Add your favorite layers here...
```
