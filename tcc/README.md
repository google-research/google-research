# Temporal Cycle-Consistency Learning (https://sites.google.com/view/temporal-cycle-consistency/home)

The codebase is useful for self-supervised representation learning on
videos. It was used in the CVPR 2019 paper Temporal Cycle-Consistency
Learning (https://arxiv.org/abs/1904.07846). Many functions will be useful for
other sequential data too.

# Self-supervised Learning Methods
Currently supported self-supervised algorithms include:

* Temporal Cycle-Consistency (algos/alignment.py)
* Shuffle and Learn (algos/sal.py)
* Time-Contrastive Networks (algos/tcn.py)
* Various combinations of the above three algorithms.
  (algos/alignment_sal_tcn.py)

We also have a supervised learning baseline that does per-frame classification
(algos/classification.py).

# Evaluation Tasks
A model that has been pre-trained with any of the above
self-supervised/supervised losses can be used for a number of downstream
fine-grained sequential/temporal understanding tasks.

We evaluated methods on 4 temporally fine-grained tasks. They are as follows:

* Phase classification (evaluation/classfication.py)
* Few-shot phase classification (evaluation/few_shot_classification.py)
* Phase progression (evaluation/event_completion.py)
* Kendall's Tau (evaluation/kendalls_tau.py)

Please refer to paper/code for definitions of these tasks.

To validate the representations, we do not fine-tune the trained models on any
of these tasks in our paper. We extract embeddings and train SVMs on top of
these embeddings. However, for practical purposes you might want to fine-tune
the pre-trained model on your task. In that case, you might find
`evaluation/algo_loss.py` useful as a skeleton that provides a loss on a given
dataset. You just need to add an optimizer to minimize this downstream loss.
Don't forget to switch set_learning_phase to 1 if you are fine-tuning.

## Preparing Data

Depending on the source of data you can use different utilities to prepare
TFRecords. The training scripts assume the data to be present in the format
described in the `decode` function in `datasets.py`.

### Videos
If you have unlabeled videos which you want to use for self-supervised
representation learning, use dataset_preparation/videos_to_tfrecords.py to
produce TFRecords.

#### Per-frame Labels for Video
If you have per-frame labels, you can run supervised learning (per-frame
classification) or use the labels for evaluation tasks. To do so, you can
use dataset_preparation/videos_to_tfrecords.py to produce TFRecords with labels
for each frame.

### Sets of images
If you have already extracted the frames of a video into images in a folder or
want to run the algorithms between sequences of images you can run use
dataset_preparation/images_to_tfrecords.py to produce TFRecords.

#### Per-frame Labels for Sets of Images
Use the fps parameter to assign a timestamp to each image in the set. Based on
this timestamp the labels will be associated with each image in the set. In case
these images are not from video, you can use an fps of 1 and timestamps at
each second to label each image.

## Training, Evaluation and Visualization
* Please download the relevant data by running this script.
`dataset_preparation/download_pouring_data.sh`

* Dowload ImageNet pre-trained ResNetV2-50 to /tmp/. If you want to download the
  checkpoint to some other location change
  CONFIG.MODEL.RESNET_PRETRAINED_WEIGHTS in config.py.
`wget -P /tmp/ https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5`

* Set directory of the library.
`root_dir=<PATH TO THIS LIBRARY ON YOUR MACHINE>

* Start training.
`python $root_dir/train --alsologtostderr`

* Start evaluation.
`python $root_dir/evaluate --alsologtostderr`

* Tensorboard.
`$tensorboard --logdir=/tmp/alignment_logs`

* Extract per-frame embeddings.
`python $root_dir/extract_embeddings --alsologtostderr`

* Visualize nearest neighbor alignments.
`python $root_dir/visualize_alignment --alsologtostderr`

## Using alignment loss on your own embeddings
If you have your own dataset of sequences, embedder (neural network), and
training code setup and only want to plugin our alignment loss on sequential
embeddings consider using functions in the library in `tcc/` folder.

To perform alignment of samples in a batch you can use the function
`compute_alignment_loss` in `tcc/alignment.py`.

To align pairs of sequences together you can use the function
`align_pair_of_sequences` in `tcc/deterministic_alignment.py`. This returns
logits and labels. To calculate the loss function itself see how the logits
and labels are used in function compute_deterministic_alignment_loss in
`tcc/deterministic_alignment.py`.

Now you can go ahead and minimize the alignment loss by optimizing over the
varibales of your embedder.

A cautionary note: please check if your embedder can encode the position of the
frame (like LSTM/ positional embeddings in Transformer). If so, then there is a
trivial solution to the TCC loss that just encodes the position of the frame in
the embedding, not learning anything semantic. The loss will go down because
the model learns to count but these embeddings might not be great for semantic
tasks like phase classification. Consider adding ways by which your embedder
finds it difficult to count.


## Citation

If you found our paper/code useful in your research, please consider citing
our paper:

`@InProceedings{Dwibedi_2019_CVPR,
author = {Dwibedi, Debidatta and Aytar, Yusuf and Tompson, Jonathan and Sermanet, Pierre and Zisserman, Andrew},
title = {Temporal Cycle-Consistency Learning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}`
