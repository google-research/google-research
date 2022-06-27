See [Resolving Referring Expressions in Images With Labeled Elements](https://arxiv.org/abs/1810.10165) for details.

Tested with tensorflow version 1.11.0

The code in the deeplab directory is copied from https://github.com/tensorflow/models/tree/master/research/deeplab

A dataset with phone screens and referring expressions is contained here:
https://github.com/google-research-datasets/uibert/tree/main/ref_exp
That is a different dataset than the one published in this paper, since the images are phone screens instead of computer screens.
It's in tf example format.
Each example has a list of bounding boxes.
The image/object/bbox/... are the coordinates of the bounding box.
The image/ref_exp/label is the index of the bounding box the referring expression refers to.
The image/ref_exp/text is the actual referring expression.
