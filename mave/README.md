# Multi-source Attribute Value Extraction (MAVE)

This repository contains code for performing benchmarks in the paper of ["MAVE: A Product Dataset for Multi-source Attribute Value Extraction"](https://dl.acm.org/doi/10.1145/3488560.3498377), published in WSDM 2022. 

## Installation 
The codebase requires Python 3. To install dependencies, we suggest to use conda virtual environment:

```
conda create -n mave python=3.10
conda activate mave
pip install -r mave/requirements.txt
```

Note that `tf-models-official==2.7.0` in `requirements.txt` is required, as of the releasing date of this repo, some functionalities of `tf-models-official` used in this repo are deprecated in higher versions.

The MAVE dataset has been released in https://github.com/google-research-datasets/MAVE, please follow instructions there for downloading the dataset. In the following, we assume [the full version of the dataset
](https://github.com/google-research-datasets/MAVE#creating-the-full-version-of-the-dataset) has been downloaded to `mave/datasets/`:

```
DATA_BASE_DIR="mave/datasets"
mkdir $DATA_BASE_DIR
```

## Splitting of dataset
We provides a binary based on [Apache Beam](https://beam.apache.org/documentation/programming-guide/) pipeline to perform various splittings of the MAVE dataset following the paper. To run the pipeline:

```
python -m mave.benchmark.data.split_json_lines_main \
--input_json_lines_filename="${DATA_BASE_DIR}/mave_positives.jsonl" \
--output_json_lines_dir="${DATA_BASE_DIR}/splits"
```

Switch `mave_positives.jsonl` to `mave_negatives.jsonl` for splitting negative examples. A nested structure of folders will be created inside `${DATA_BASE_DIR}/splits` containing various kinds of splits discussed in the paper. Please see `mave/data/datasets.py` for specifications for all splitted datasets.

## Converting to TFRecords
This following example shows how to convert to TFRecords for the splitted datasets.
First we download BERT vocab file from BERT [uncased_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) checkpoint to `BERT_VOCAB_FILE`, and ETC vocab file from ETC [etc_base_2x_pretrain](https://storage.googleapis.com/gresearch/etcmodel/checkpoints/etc_base_2x_pretrain.zip) checkpoint to `ETC_VOCAB_FILE`, then run

```
# BERT model format.
python -m mave.benchmark.data.create_tf_records_main \
--config="mave/benchmark/configs.py" \
--config.bert.vocab_file="${BERT_VOCAB_FILE}" \
--config.etc.vocab_file="${ETC_VOCAB_FILE}" \
--model_type="bert" \
--input_json_lines_filepattern="${DATA_BASE_DIR}/splits/PRODUCT/*/*/mave_*.jsonl"
```

`model_type` must be one of "bert", "etc", or "bilstm_crf". Similarly, we need to convert for all datasets specified in `mave/data/datasets.py` by changing `model_type` and `input_json_lines_filepattern`. The TFRecords will be stored in same directories as corresponding JSON Line files in `${DATA_BASE_DIR}/splits/`.

## Model training
The following example shows how to train and evaluate a BERT tagger model using the splitted datasets. First we download `bert_config.json` from BERT [uncased_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) checkpoint to `BERT_CONFIG_FILE`, and `etc_config.json` from ETC [etc_base_2x_pretrain](https://storage.googleapis.com/gresearch/etcmodel/checkpoints/etc_base_2x_pretrain.zip) checkpoint to `ETC_CONFIG_FILE`, then run

```
python -m mave.benchmark.run_main \
--config="mave/benchmark/configs.py" \
--config.model_type="bert" \
--config.bert.bert_config_file="${BERT_CONFIG_FILE}" \
--config.etc.etc_config_file="${ETC_CONFIG_FILE}" \
--config.data.version="00_All_bert" \
--config.bert.bert_hub_module_url="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4" \
--config.train.model_dir="mave/experiments/0" \
--config.train.num_train_steps=10 \
--config.train.steps_per_epoch=2 \
--config.train.steps_per_loop=2 \
--config.train.save_summary_steps=2 \
--config.train.num_warmup_steps=2 \
--config.train.train_batch_size=1 \
--config.train.save_checkpoints_steps=2 \
--use_tpu=False \
--tpu=''
```

To use [`tf.distribute.TPUStrategy`](https://www.tensorflow.org/guide/distributed_training#tpustrategy), `use_tpu=True` and `tpu` (TPU address) need to be set.

More details on configs and datasets can be found in `mave/configs.py` and `mave/data/datasets.py`, which contain actual hyperparameters used in the paper.

In the paper, we used the following pretrained checkpoints:
##### BiLSTM-CRF
Word embeddings initialized from TF Hub checkpoint:

- https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4

##### AVEQA
BERT checkpoint TF Hub urls:

- https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2 \
- https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2 \
- https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2 \
- https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/2 \
- https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4 \
- https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4

##### MAVEQA
ETC checkpoint download link:

- [etc_base_2x_pretrain](https://storage.googleapis.com/gresearch/etcmodel/checkpoints/etc_base_2x_pretrain.zip)

## Inference
The following example shows how to run batch inference using trained checkpoint on certain eval datasets:

```
python -m mave.benchmark.run_inference \
--config="mave/benchmark/configs.py" \
--model_type="bert" \
--saved_model_dir="mave/experiments/0/exported_models" \
--input_json_lines_filepattern="${DATA_BASE_DIR}/splits/PRODUCT/eval/00_All/mave_*.jsonl"
```

Similarly, we can run inference using other models on other datasets by changing `model_type`, `saved_model_dir`, and `input_json_lines_filepattern`. More options see flags in `run_inference.py`. 

## Bibtex
To cite this work, please use:

```bibtex
@inproceedings{liyang2022mave,
title = {MAVE: A Product Dataset for Multi-Source Attribute Value Extraction},
author = {Yang, Li and Wang, Qifan and Yu, Zac and Kulkarni, Anand and Sanghai, Sumit and Shu, Bin and Elsas, Jon and Kanagal, Bhargav},
booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
year = {2022},
pages = {1256–1265},
series = {WSDM '22},
url = {https://doi.org/10.1145/3488560.3498377},
doi = {10.1145/3488560.3498377},
}
```

*This is not an officially supported Google product.*
