# Axial Transformers

This repository contains an implementation of "Axial Attention in Multidimensional Transformers" (https://arxiv.org/abs/1912.12180), by Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans (first two authors contributed equally).

## Instructions

```
# Specify the desired model/dataset
export MODEL_NAME=imagenet32  # or imagenet64

# Download model checkpoint
mkdir logdir
gsutil cp gs://axial-transformers/$MODEL_NAME/{checkpoint,model.ckpt.data-00000-of-00001,model.ckpt.index,model.ckpt.meta} logdir

# Run the model
python3 -m axial.main --config $MODEL_NAME --logdir logdir

# A TensorBoard log will be written to logdir, including samples from the model.
```
