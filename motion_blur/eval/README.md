# Learning to Synthesize Motion Blur

This subset of our code release is for evaluating a trained model on the test
set presented in this paper. Because we do not provide our trained model or our
synthetic training set in this code release, we instead provide the output of
our trained model on this test set. Running this code on a newly trained model
will require that the user modifies this script to load and run a trained TF
graph, or runs their model on this test set elsewhere and writes the output
of evaluation to disk.

First, download [the dataset we use for evaluation](https://drive.google.com/file/d/1AcxuWl2PnqkyxyTgmzepwqw48G15rT2U/view?usp=sharing),
and extract it to `./test_dataset/`. Then, download [the output of our model](https://drive.google.com/file/d/1WmpPGV2iGU6MNlwEopxANAXuyMhnC675/view?usp=sharing),
and extract it to `./models/`. This folder contains the output of three trained models used in the paper:

* `./models/motion_blur/` corresponds to the complete model proposed in the paper,
* `./models/motion_blur_direct/` corresponds to the "Direct Prediction" ablation.
* `./models/motion_blur_uniform/` corresponds to the "Uniform Weight" ablation.

Run `evaluate.py` to evaluate the performance of a trained model on the motion blur test dataset.
