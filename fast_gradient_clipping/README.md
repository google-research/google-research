# Fast Gradient Clipping

This repository is the official implementation of the NeurIPS 2023 paper
[A Unified Fast Gradient Clipping Framework for DP-SGD](https://neurips.cc/virtual/2023/poster/70754).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running in a Virtual Environment

The shell script `run.sh` includes an example of how to run the small `fc0.py`
experiment in a Python virtual environment. It should be called from the
`google_research` directory, i.e., call

```test
./fast_gradient_clipping/run.sh
```

## Experiments

To replicate the experiments profiling runtime, run the scripts the `scripts`
directory by calling `python -m [target]`. These scripts will generate plots as
`*.svg` files in your `/tmp/` directory.

Unfortunately, the experiments profiling peak memory usage require libraries
that are internal to Google. Users can add their own memory profiling tool by
appropriately modifying the following functions in `./src/profiling_tools.py`:

```profile
get_compute_profile()
get_compute_profile_with_vocab()
get_train_bert_model_compute_profile()

```

For reference, the full list of relevant targets, and their corresponding
experiments are listed below.

&nbsp;

**Target** | **Experiment**
---------- | ---------------------------------
`fc0.py`   | Small Test Script
`fc1.py`   | Fully Connected Layer (Main Body)
`fc2.py`   | Fully Connected Layer (Appendix)
`emb1.py`  | Embedding Layer
`bert1.py` | BERT Model
