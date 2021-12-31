# Felix

Felix is flexible text-editing approach for generation, designed to derive
maximum benefit from the ideas of decoding with bi-directional contexts and
self-supervised pretraining. We achieve this by decomposing the text-editing
task into two sub-tasks: **tagging** to decide on the subset of input tokens and
their order in the output text and **insertion** to in-fill the missing tokens in
the output not present in the input.



A detailed method description and evaluation can be found in our EMNLP2020 findings paper:
[https://www.aclweb.org/anthology/2020.findings-emnlp.111/](https://www.aclweb.org/anthology/2020.findings-emnlp.111/)

Felix is built on Python 3, Tensorflow 2 and
[BERT](https://github.com/tensorflow/models/tree/master/official/nlp/bert). It works with CPU, GPU, and
Cloud TPU.

## Usage Instructions

Running an experiment with Felix consists of the following steps:

1. Create label_map for tagging model
2. Convert data for insertion/tagging model.
3. Finetune the tagging/insertion models.
4. Compute predictions.


Next we go through these steps, using a subset of DiscoFuse
([DiscoFuse](https://github.com/google-research-datasets/discofuse)) task as a
running example.

You can run all of the steps with

```
sh run_discofuse_experiment.sh
```

After setting the variables in the beginning of the script.

### 1. Label map construction


```
# Label map construction
export OUTPUT_DIR=/path/to/output

python phrase_vocabulary_constructor_main \
--output="${OUTPUT_DIR}/label_map.json" \
--use_pointing="${USE_POINTING}" \
--do_lower_case="True"

```

### 2. Converting data for insertion/tagging model

Download a pretrained BERT model from the
[official repository](https://github.com/tensorflow/models/tree/master/official/nlp/bert#access-to-pretrained-checkpoints).
We've used the 12-layer (NOT Pretrained hub modules) [''BERT-Base'' model](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12.tar.gz) for all
of our experiments,unless otherwise stated. Then convert the original TSV
datasets into TFRecord format. The discofuse dataset can be found here(https://github.com/google-research-datasets/discofuse)

```
# Preprocess
export BERT_BASE_DIR=/path/to/uncased_L-12_H-768_A-12
export DISCOFUSE_DIR=/path/to/discofuse

python preprocess_main \
  --input_file="${DISCOFUSE_DIR}/train.tsv" \
  --input_format="discofuse" \
  --output_file="${OUTPUT_DIR}/train.tfrecord" \
  --label_map_file="${OUTPUT_DIR}/label_map.json" \
  --vocab_file="${BERT_BASE_DIR}/vocab.txt" \
  --do_lower_case="True" \
  --use_open_vocab="True" \
  --max_seq_length="128" \
  --use_pointing="${USE_POINTING}" \
  --split_on_punc="True"

python preprocess_main.py \
  --input_file="${DISCOFUSE_DIR}/tune.tsv" \
  --input_format="discofuse" \
  --output_file="${OUTPUT_DIR}/tune.tfrecord" \
  --label_map_file="${OUTPUT_DIR}/label_map.json" \
  --vocab_file="${BERT_BASE_DIR}/vocab.txt" \
  --do_lower_case="True" \
  --use_open_vocab="True" \
  --max_seq_length="128" \
  --use_pointing="${USE_POINTING}" \
  --split_on_punc="True"
```

### 3. Model Training

Model hyperparameters are specified in [felix_config.json](discofuse/felix_config.json). This configuration file extends
`bert_config.json` which comes with the zipped pretrained BERT model.
**note** These models can be trained independently, as such it is quicker to train them in parallel rather than sequentially.


Train the models on CPU/GPU.

```
# Train
python run_felix \
    --train_file="${OUTPUT_DIR}/train.tfrecord" \
    --eval_file="${OUTPUT_DIR}/tune.tfrecord" \
    --model_dir_tagging="${OUTPUT_DIR}/model_tagging" \
    --bert_config_tagging="${DISCOFUSE_DIR}/felix_config.json" \
    --max_seq_length=128 \
    --num_train_epochs=500 \
    --num_train_examples=8 \
    --num_eval_examples=8 \
    --train_batch_size="32" \
    --eval_batch_size="32" \
    --log_steps="100" \
    --steps_per_loop="100" \
    --train_insertion="False" \
    --use_pointing="${USE_POINTING}" \
    --init_checkpoint="${BERT_DIR}/bert_model.ckpt" \
    --learning_rate="0.00003" \
    --pointing_weight="1" \
    --input_format="recordio" \
    --use_weighted_labels="True"

rm -rf "${DATA_DIRECTORY}/model_insertion"
mkdir "${DATA_DIRECTORY}/model_insertion"
python run_felix \
    --train_file="${OUTPUT_DIR}/train.tfrecord.ins" \
    --eval_file="${OUTPUT_DIR}/tune.tfrecord.ins" \
    --model_dir_insertion="${OUTPUT_DIR}/model_insertion" \
    --bert_config_insertion="${DISCOFUSE_DIR}/felix_config.json" \
    --max_seq_length=128 \
    --num_train_epochs=500 \
    --num_train_examples=8 \
    --num_eval_examples=8 \
    --train_batch_size="32" \
    --eval_batch_size="32" \
    --log_steps="100" \
    --steps_per_loop="100" \
    --init_checkpoint="${BERT_DIR}/bert_model.ckpt" \
    --use_pointing="${USE_POINTING}" \
    --learning_rate="0.00003" \
    --pointing_weight="1" \
    --input_format="recordio" \
    --train_insertion="True"
```

To train on Cloud TPU, you should additionally set:

```
  --use_tpu=true \
  --tpu_name=${TPU_NAME}
```

Please see [BERT TPU instructions](https://github.com/google-research/bert#fine-tuning-with-cloud-tpus) and the
[Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist)
for how to use Cloud TPUs.

### 4. Prediction


```
# Predict
export PREDICTION_FILE=${OUTPUT_DIR}/pred.tsv

python predict_main \
--input_format="discofuse" \
--predict_input_file="${DISCOFUSE_DIR}/test.tsv" \
--predict_output_file="${PREDICTION_FILE}"\
--label_map_file="${OUTPUT_DIR}/label_map.json" \
--vocab_file="${BERT_BASE_DIR}/vocab.txt" \
--max_seq_length=128 \
--predict_batch_size=32 \
--do_lower_case="True" \
--use_open_vocab="True" \
--bert_config_tagging="${DISCOFUSE_DIR}/felix_config.json" \
--bert_config_insertion="${DISCOFUSE_DIR}/felix_config.json" \
--model_tagging_filepath="${OUTPUT_DIR}/model_tagging" \
--model_insertion_filepath="${OUTPUT_DIR}/model_insertion" \
--use_pointing="${USE_POINTING}"
```

To predict on Cloud TPU, you should additionally set:

```
  --use_tpu=true \
  --tpu_name=${TPU_NAME}
```

The predictions output a TSV file with four columns: Source, the input to the insertion model, the final output, and the reference. Note the felix output is tokenized (WordPieces), including a start "[CLS]" and end "[SEP]". WordPieces can be removed by replacing " ##" with "". Additionally words have been split on punctuation "don't -> don ' t", this must also be reversed.

## How to Cite Felix

```
@inproceedings{mallinson-etal-2020-felix,
    title = '{FELIX}: Flexible Text Editing Through Tagging and Insertion',
    author = 'Mallinson, Jonathan  and
      Severyn, Aliaksei  and
      Malmi, Eric  and
      Garrido, Guillermo',
    booktitle = 'Findings of the Association for Computational Linguistics: EMNLP 2020',
    month = nov,
    year = '2020',
    address = 'Online',
    publisher = 'Association for Computational Linguistics',
    url = 'https://www.aclweb.org/anthology/2020.findings-emnlp.111',
    doi = '10.18653/v1/2020.findings-emnlp.111',
    pages = '1244--1255',
 }
```

## License

Apache 2.0; see [LICENSE](LICENSE) for details.

## Disclaimer

This repository contains a TensorFlow 2 reimplementation of our original
TensorFlow 1 code used for the paper and thus some discrepancies compared to the paper
results are possible. However, we've verified that we get the similar results on the DiscoFuse dataset.

This is not an official Google product.
