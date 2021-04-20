# Compression Image Quality Assessment Dataset (CIQA)


## Overview

The CIQA dataset is an open-sourced collection of labels to the popular
[Aesthetic Visual Analysis (AVA)] (https://paperswithcode.com/dataset/aesthetic-visual-analysis)
dataset. Images from AVA were sampled, compressed to different JPEG quality
factors using the Tensorflow tf.image.adjust_jpeg_quality method and rated by
human raters in a forced choice pairwise comparison study.
This dataset is the result of the work in [Deep Perceptual Image Quality
Assessment for Compression] (https://arxiv.org/abs/2103.01114)

## AVA Image Sampling

The [Aesthetic Visual Analysis dataset (AVA)] (https://ieeexplore.ieee.org/document/6247954)
is well suited for deep learning applied to aesthetic Image Quality Assessment
(IQA) proven by its successful implementation in the
[NIMA] (https://arxiv.org/abs/1709.05424) model.
AVA contains ∼ 255,000 images rated based on aesthetic qualities by amateur
photographers. Because these images are stored using JPEG compression, only the
subset of images with a JPEG quality factor, Q, of 99 or more (near lossless)
were sampled using the ImageMagick imaging tool software.

## Sampled Image Compression
The sampled images are compressed at two random
JPEG quality factors. The AVA dataset contains semantic labels in addition
to perceptual quality rating. It is worth noting our sampling preserves the
distribution of semantic classes. Semantic labels for each class of animal,
architecture, cityscape, floral, food/drink, landscape, portrait, and still life
have almost equal occurrence (∼ 6.25%). All reference images and the
category of ’generic’ has an occurrence of 50%. This was done to ensure the
diversity of sampled images and to create a more accurate representation of 
general perceptual image quality. Similarly, we preserve a wide distribution of
resolutions from the sampled images which which varies from 200 × 200 to
800 × 800, and not necessarily always of equal height and width.


## Labels
7808 pairwise comparisons were generated and each was rated by 32 individual
participants. Pairs were chosen by compressing the reference image to two random
JPEG quality factors from 10 to 100. 13,868 compressed images were generated
from the 6,667 reference images. Of the 6,667 images sampled 6,372 reference
images were used in the training set and 256 reference images were used in the
training set. Each image in the training set was compressed with two different
JPEG quality factors producing 1 pairwise comparison for each image and 6,372
total training examples. Each image in the test set was compressed with 4
different JPEG quality factors producing 6 pairwise comparisons for each
reference image in the test set and 1536 total examples in the test set.

## Resulting Dataset
The result is a dataset with a training set (6372 pairwise comparisons) and a
test set (1536 pairwise comparisons). Column names in the data are
* ref_id: The reference image ID from the AVA dataset
* jpg1: This field is the first image's name formatted as
{reference_image_id}/{jpg_Quality_Factor_1}.
* jpg1: This field is the second image's name formatted as
{reference_image_id}/{jpg_Quality_Factor_2}.
* q1: The quality factor used to compress the first image from the reference image
* q2: The quality factor used to compress the second image from the reference image
* jpg2_pref: The label from 0.0 to 1.0 corresponding to the proportion of raters that prefered the second image. The proportion of raters who prefer the first image can be inferred as 1 - jpg2+pref

## Access
The train and test sets are stored in the Google Research GCP public data
storage. The data can be accessed through the gsutil CLI, tf.io.gfile API and
HTTP api. It is stored in gs://gresearch bucket under the CIQA directory.

* Training Set - http://storage.googleapis.com/gresearch/ciqa/train.csv
* Test Set - http://storage.googleapis.com/gresearch/ciqa/test.csv
