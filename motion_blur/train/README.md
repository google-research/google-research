# Learning to Synthesize Motion Blur

This code implements the training of our model. This code release does not
include the synthetic training data we use, so the user must provide their own
training data.

The code is implemented in Tensorflow and the required packages are listed in
`requirements.txt`. To train a motion blur model, run `train.py`, setting
`--model_dir` to the output directory at which to save the model, and
`--train_pattern` and `--test_pattern` to patterns pointing to source
directories containing sharp and motion blurred images that can be used for
train and test data respectively. Patterns may include wildcards, such as
`path/to/train/dirs/*/*`. See `dataset.read_images()` for how the training data
is expected to be structured.
