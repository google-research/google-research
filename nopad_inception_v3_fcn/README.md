Inception v3 in Fully Convolutional Network mode used in the following paper:
[An Augmented Reality Microscope with Real-time Artificial Intelligence Integration for Cancer Diagnosis](https://www.nature.com/articles/s41591-019-0539-7).

This is a variant of inception v3 by removing all the paddings and adopting the
principle of fully convolutional network design. This change allows the network
to be trained and inference run with different patch size while having the same
inference results.

This folder can be downloaded using subversion:
```svn export https://github.com/google-research/google-research/trunk/nopad_inception_v3_fcn```.

An example uptraining script is provided at ```finetune_on_flowers.sh```. Note
the setup instructions detailed in the file.
