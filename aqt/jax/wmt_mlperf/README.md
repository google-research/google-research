## Machine Translation

Trains a Transformer-based model (Vaswani *et al.*, 2017) on the WMT Machine
Translation en-de dataset.

This model can be optionally quantized with AQT style
quantization. This example uses linear learning rate warmup and inverse square
root learningrate schedule.

### Requirements

*   TensorFlow datasets `wmt17_translate/de-en` and `wmt14_translate/de-en` need
    to be downloaded and prepared. A sentencepiece tokenizer vocabulary will be
    automatically generated and saved on each training run.
*   This example additionally depends on the `sentencepiece` and
    `tensorflow-text` packages.


### How to run on Cloud TPU

Setup the TPU VM and install the dependencies on it as described
[here](https://cloud.google.com/tpu/docs/jax-pods) for creating pod slices, or
[here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm) for a single
v3-8 TPU.

First create a single TPUv3-8 VM and connect to it (you can find more detailed
instructions [here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)):

```
ZONE=us-central1-a
TPU_TYPE=v3-8
TPU_NAME=$USER-flax-wmt

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --zone $ZONE \
    --accelerator-type $TPU_TYPE \
    --version v2-alpha

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE -- \
    -L 6006:localhost:6006
```

When connected install JAX:

```
pip install "jax[tpu]>=0.2.7" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Then install other dependencies:

```
pip install -r requirements.txt
```

And finally start the training:

```
python3 train.py --model_dir /tmp/wmt_w4_a4_auto \
--hparams_config_dict hparams_configs/leaderboard/full_model_4bit_weights_and_auto_acts.py \
--jax_backend_target="grpc://192.168.0.2:8470"

```

Note that you might want to set `TFDS_DATA_DIR` as explained below. You probably
also want to start the long-running command above in a `tmux` session and start
some monitoring in a separate pane (note that we forwarded port 6006 locally
above):

```
tensorboard --logdir=$HOME/logs
```

### Acceptable quantization options

Currently, all Matmul layers (all `DenseGeneral` layers and the `Dense` layer)
have quantization support.

### Downloading the WMT Datasets

We recommend downloading and preparing the TFDS datasets beforehand. For Cloud
TPUs, we recommend using a cheap standard instance and saving the prepared TFDS
data on a storage bucket, from where it can be loaded directly. Set the
`TFDS_DATA_DIR` to your storage bucket path (`gs://<bucket name>`).

You can download and prepare any of the WMT datasets using TFDS directly:
`python -m tensorflow_datasets.scripts.download_and_prepare
--datasets=wmt17_translate/de-en`

The typical academic BLEU evaluation also uses the WMT 2014 Test set: `python -m
tensorflow_datasets.scripts.download_and_prepare
--datasets=wmt14_translate/de-en`
