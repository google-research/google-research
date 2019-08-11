Inception v3 in Fully Convolutional Network mode used in the following paper.

An Augmented Reality Microscope with Real-time Artificial Intelligence Integration for Cancer Diagnosis
link: TBA.

This is a variant of inception v3 by removing all the paddings and adopting the
principle of fully convolutional network design. This change allows the network
to be trained and inference run with different patch size while having the same
inference results. The training loop will train the model for 20 steps on mnist
dataset and then stop. To use this for histopathology images, please set up
proper training and evaluation loops.
