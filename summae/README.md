## Overview

This directory contains code for generating the data and model described in
"SummAE: Zero-Shot Abstractive Text Summarization using Length-Agnostic Auto-Encoders"

Disclaimer: This is not an official Google product.

## Install dependencies (`run.sh`)
This package depends on Tensorflow and google_research/rouge. See
the included `run.sh` for how to install and run a unit test inside
of a `virtualenv`.

## Generate rocstories data.
Request (free) and download raw data for [ROCStories corpora](http://cs.rochester.edu/nlp/rocstories/)
into a directory pointed to by the environment variable, and run the
data processing script:
```
export ROCSTORIES_RAW=absoluatepath/raw_rocstories
# Download raw rocstories data to $ROCSTORIES_RAW
export ROCSTORIES_DATA=absolutepathto/processed_rocstoriesdata
# For example, inside google_research/google_research directory:
bash summae/generate_data.sh $ROCSTORIES_RAW summae/testdata $ROCSTORIES_DATA
```

### Verify data
```
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
bash run_locally.sh decode /tmp/testmodel 0
```

### Run decode of best model
```
mkdir /tmp/best
cp -r summae/testdata/best /tmp/best
bash run_locally.sh decode /tmp/best 358000
```
