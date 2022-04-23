# Inpainting and Neighborhood Techniques for Prediction of Cognate Reflexes

This directory contains the software developed for the
[SIGTYP 2022 Shared Task](https://github.com/sigtyp/ST2022) on prediction of
cognate reflexes held at the The Fourth Workshop on Computational Typology and
Multilingual NLP (SIGTYP 2022).

There are two model implementations in this directory:

1.  Cognate Neighbors model (under `neighbors` subdirectory).
1.  Cognate Inpainting model (under `inpaint` subdirectory).

## Dependencies

Clone the data from the official shared task repository:

```shell
git clone https://github.com/sigtyp/ST2022.git
```

Please check [requirements.txt](requirements.txt) for a list of basic
dependencies for most of the tools in this directory. To install the
dependencies in the Python virtual environment run

```shell
pip3 install -r requirements
```


## Cognate Neighbors

In the following examples we are preparing the data, training and decoding the
model in case of `Zway` language from the Ethiosemitic group.


### Preparing the data

Prepare the training data: Given the SIGTYP shared task data in
`${OUTPUT_DATA_DIR}` preprocess and augment it with randomly generated
language/pronunciation pairs, and place the output in `${OUTPUT_DATA_DIR}`. The
preprocessed data will be in binary format
([TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)) that is
going to be consumed by the [Tensorflow](https://www.tensorflow.org/) model
trainer described below:

```shell
  DATA_DIR=~/projects/ST2022/data
  OUTPUT_DATA_DIR=/tmp/tmp
  python neighbors/data/create_neighborhood.py \
    --task_data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DATA_DIR} \
    --max_rand_len 10 \
    --language_group felekesemitic \
    --lang Zway \
    --logtostderr
```

The above command will generate three files in the `${OUTPUT_DATA_DIR}`
directory:

*  Training data in `Zway_train.tfrecords` (572,000 records) and test data in
   `Zway_test.tfrecords` (34 records).
*  Combined symbol table `Zway.syms` in text format corresponding to the
   characters from language names and IPA phonemes from the corresponding
   pronunciation strings. This table contains 92 symbols that include special
   symbols such as beginning and end-of-sentence, and sequence padding markers.

Please note: pronunciation sequences are assumed to be sequences of
space-delimited phonemes and are split into individual elements. Languages names
are kept as is.

### Training

The training pipeline is implemented in
[Lingvo](https://github.com/tensorflow/lingvo) which is a framework for building
neural networks in Tensorflow, particularly sequence models. To run the training
pipeline for the tiny Transformer configuration `TransformerWithNeighborsTiny`:

```shell
  MODEL_TRAIN_DIR=/tmp/logdir
  > python neighbors/model/trainer.py \
    --run_locally=cpu \
    --model=feature_neighborhood_model_config.TransformerWithNeighborsTiny \
    --input_symbols ${OUTPUT_DATA_DIR}/Zway.syms \
    --output_symbols ${OUTPUT_DATA_DIR}/Zway.syms \
    --feature_neighborhood_train_path tfrecord:${OUTPUT_DATA_DIR}/Zway_train.tfrecords \
    --feature_neighborhood_dev_path tfrecord:${OUTPUT_DATA_DIR}/Zway_test.tfrecords \
    --feature_neighborhood_test_path tfrecord:${OUTPUT_DATA_DIR}/Zway_test.tfrecords \
    --max_neighbors=18 --max_pronunciation_len=13 --max_spelling_len=36 \
    --logdir ${MODEL_TRAIN_DIR}
```

The above command will run the pipeline training the model checkpoints in
`${MODEL_TRAIN_DIR}` using the training, development and test splits (test and
development splits point to the same dataset) in `TFRecord` format. The input
and output symbols for the "tiny" Transformer model are taken from the combined
symbol table.

Please note the `--run_locally=cpu` flag - it requests the training to be
performed using the CPUs on a local machine. Even though the model is "tiny", it
still takes over day and a half to train. During the training, the best (so far)
Tensorflow model checkpoints along with other information are saved
to `${MODEL_TRAIN_DIR}`.

### Decoding

When the Tensorflow model checkpoints are available under
`${MODEL_TRAIN_DIR}/train`, evaluation of these checkpoints can be performed as
follows:

```shell
  RESULTS_DIR=/tmp/tmp/decode_dir
  python neighbors/model/decoder.py \
    --feature_neighborhood_test_path=tfrecord:${OUTPUT_DATA_DIR}/Zway_test.tfrecords \
    --input_symbols ${OUTPUT_DATA_DIR}/Zway.syms \
    --output_symbols ${OUTPUT_DATA_DIR}/Zway.syms \
    --ckpt ${MODEL_TRAIN_DIR}/train \
    --batch_size 1 \
    --num_samples 1000 \
    --model TransformerWithNeighborsTiny \
    --decode_dir ${RESULTS_DIR} \
    --split_output_on_space \
    --max_neighbors=18 --max_pronunciation_len=13 --max_spelling_len=36
```

The above command will pick the latest model checkpoint from
`${MODEL_TRAIN_DIR}/train` directory and evaluate it against the test data
writing the resulting files to `${RESULTS_DIR}`:

*  The detailed evaluation results for each example in `decode_-1.txt`.
*  The neighborhood attention matrix for visualization in
   `neighbor_attention.txt`.
*  The results summary in `results_-1.txt`.

The model outputs correspond to pronunciations and these are split on space by
the decoder using `--split_output_on_space` flag to match the corresponding
operation during the data preparation stage.

The `-1` in file names above refers to the checkpoint limit, set to `-1` by
default if `--ckpt_limit` flag is unspecified in the decoder. This basically
means that the latest model checkpoint was evaluated.

## Cognate Inpainting

The model is inspired by image infill/inpainting
[work](https://arxiv.org/pdf/1804.07723.pdf) by NVidia. There, random masks are
applied to images to blank out certain pixels, and a convolutional network is
used to restore those pixels given the surrounding context.

TODO: Complete.

### Training

```shell
  DATA_DIR=~/projects/ST2022
  MODEL_TRAIN_DIR=/tmp/logdir
  python inpaint/cognate_inpaint.py \
    --data_dir ${DATA_DIR} \
    --checkpoint_dir ${MODEL_TRAIN_DIR} \
    --max_epochs 100
```

### Inference

```shell
  RESULTS_FILE=/tmp/results-0.10.tsv
  python inpaint/cognate_inpaint.py \
    --data_dir ${DATA_DIR} \
    --checkpoint_dir ${MODEL_TRAIN_DIR} \
    --decode \
    --output_results_tsv ${RESULTS_FILE}
```
