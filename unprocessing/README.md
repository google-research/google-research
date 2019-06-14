# Unprocessing Images for Learned Raw Denoising

Reference code for the paper [Unprocessing Images for Learned Raw Denoising](http://timothybrooks.com/tech/unprocessing).
Tim Brooks, Ben Mildenhall, Tianfan Xue, Jiawen Chen, Dillon Sharlet, Jonathan T. Barron
CVPR 2019

Please cite the article above if you publish results obtained with this code.

The code is implemented in Tensorflow and the required packages are listed in `requirements.txt`. To train an unprocessing model, run `train.py`, setting `--model_dir` to the output directory at which to save the model, and `--train_pattern` and `--test_pattern` to patterns pointing to source JPG images that can be used to generate train and test data respectively. Patterns may include wildcards, such as `path/to/train/images/*`. Any large dataset of unlabeled JPG images or image patches, such as MIRFlickr, can be used as source data.
