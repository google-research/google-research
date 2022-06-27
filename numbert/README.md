
# NumBERT


This repository contains the code for creating the NUMBert Language model 
described in: 
[Do Language Embeddings Capture Scales?](https://arxiv.org/abs/2010.05345).

We also share the model itself.

## Pre-training

NUMBert is a modfication of BERT that introduces the use of Scientific notation
(e.g. 137.25-> 13725E2) to represent numbers. In the first step of the LM
training, we preprocess the training data corpus by converting all instances of 
numbers into scientific notation. We introduce a new wordpeice token 
"scinotexp" to represent the exponent symbol. The rest of the training pipeline
remains identical to the original BERT implementation. To prepare the
pretraining data, you can use the following invocation:

```shell
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=model/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

See the [BERT repository](https://github.com/google-research/bert) for more 
information. Note that the vocab file has been modified to introduce the 
"scinotexp" token and has been included in this release.

## Model
We also release a NumBERT model trained using the standard BERT pre-training
pipeline with the modifications described above. The files are 
[here](https://console.cloud.google.com/storage/browser/gresearch/numbert/).

## Data

The data for the probing experiments are taken from the 
[Distributions over Quantities](https://github.com/google-research-datasets/distribution-over-quantities) 
resource. In addition, evaluations were performed on 2 datasets:
[VerbPhysics](https://uwnlp.github.io/verbphysics/) and the 
[2018 version of the Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html).
