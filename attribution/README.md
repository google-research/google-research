# Tensorflow Ops for Deep Neural Network Attribution.
Uses the Integrated Gradients Technique from the paper:
https://arxiv.org/abs/1703.01365


This directory contains functions for augmenting an existing tensorflow graph in-place to add ops for computing Integrated Gradients.
It uses the Tensorflow Graph Editor to add ops that perform the entire Integrated Gradient computation within a single sess.run call.
It works for both continuous inputs, like images, and discrete inputs with embeddings. Example usage of the functions can be found in the test cases.
