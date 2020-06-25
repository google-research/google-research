Disclaimer: This is not an official Google product.

## Overview

This directory contains code for generating the data and model described in
["SummAE: Zero-Shot Abstractive Text Summarization using Length-Agnostic Auto-Encoders"](http://arxiv.org/abs/1910.00998).

For questions or issues, contact peterjliu@google.com.

## Install dependencies (`run.sh`)
This package depends on Tensorflow and google_research/rouge. See
the included `run.sh` for how to install and run a unit test inside
of a `virtualenv`.

## Generate rocstories data.
Request (free) and download raw data for [ROCStories corpora](http://cs.rochester.edu/nlp/rocstories/)
into a directory pointed to by the environment variable `ROCSTORIES_RAW`.
This directory should contain the following files: 
* "ROCStories_winter2017 - ROCStories_winter2017.csv"
* "ROCStories__spring2016 - ROCStories_spring2016.csv"

Then run the data processing script:

```bash
export ROCSTORIES_RAW=absolutepathto/raw_rocstories
export ROCSTORIES_DATA=absolutepathto/processed_rocstoriesdata
```

Inside google_research/google_research directory:

```bash
bash summae/generate_data.sh $ROCSTORIES_RAW summae/testdata $ROCSTORIES_DATA
```

### Verify data
```bash
python -m summae.verify_data --data_dir=$ROCSTORIES_DATA
```

## Running the model code
### Train from scratch for a few steps using `run_locally.sh`
```
export HYPERS=`pwd`/testdata/hypers.json
bash summae/run_locally.sh train /tmp/testmodel
```

### Decode latest model checkpoint
```
bash summae/run_locally.sh decode /tmp/testmodel 0
```

### Run decode of best model
```
mkdir /tmp/best
wget -P /tmp/best https://storage.googleapis.com/gresearch/summae/testdata/best/checkpoint
wget -P /tmp/best https://storage.googleapis.com/gresearch/summae/testdata/best/model.ckpt-358000.data-00000-of-00001
wget -P /tmp/best https://storage.googleapis.com/gresearch/summae/testdata/best/model.ckpt-358000.index
wget -P /tmp/best https://storage.googleapis.com/gresearch/summae/testdata/best/model.ckpt-358000.meta
bash summae/run_locally.sh decode /tmp/best 358000
```

Decodes output is saved to `/tmp/best/decodes/`.

## Citation
If you use this code in research please cite:
```
@article{liu2019summae,
  title={SummAE: Zero-Shot Abstractive Text Summarization using Length-Agnostic
  Auto-Encoders},
  author={Liu,  Peter J. and Chung, Yu-An and Ren, Jie},
  journal={arXiv preprint arXiv:1910.00998},
  url={http://arxiv.org/abs/1910.00998},
  year={2019}
}
```
##
