# Inpainting and Neighborhood Techniques for Prediction of Cognate Reflexes

This directory contains the software developed for the [SIGTYP 2022 Shared
Task](https://github.com/sigtyp/ST2022) on prediction of cognate reflexes held
at the 4th Workshop on Research in Computational Typology and Multilingual NLP
([SIGTYP 2022](https://sigtyp.github.io/workshop.html)).

There are two model implementations in this directory:

1.  Cognate Neighbors model (under the `neighbors` subdirectory). This model
    corresponds to two systems submitted to the shared task:
    *   Model for each individual language. The configurations
        [mockingbird-n1-a](https://github.com/sigtyp/ST2022/tree/main/systems/mockingbird-n1-a),
        [mockingbird-n1-b](https://github.com/sigtyp/ST2022/tree/main/systems/mockingbird-n1-b),
        and
        [mockingbird-n1-c](https://github.com/sigtyp/ST2022/tree/main/systems/mockingbird-n1-c)
        correspond to the models trained for 25000, 30000 and 100000 steps,
        respectively.
    *   Model for each language group:
        [mockingbird-n2](https://github.com/sigtyp/ST2022/tree/main/systems/mockingbird-n2). The
        stopping condition ad-hoc corresponding to varying numbers of training steps.
1.  Cognate Inpainting model (under the `inpaint` subdirectory):
    [mockingbird-i1](https://github.com/sigtyp/ST2022/tree/main/systems/mockingbird-i1).

## Dependencies

Checkout this codebase (you'll need `svn` to checkout this individual
directory):

```shell
svn export https://github.com/google-research/google-research/trunk/cognate_inpaint_neighbors
```

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


## Cognate Neighbors Model

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
  DATA_DIR=~/projects/ST2022
  OUTPUT_DATA_DIR=/tmp/tmp
  python neighbors/data/create_neighborhood.py \
    --task_data_dir ${DATA_DIR}/data \
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
  RESULTS_DIR=${OUTPUT_DATA_DIR}/decode_dir
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
*  The results in SIGTYP Shared task format: `results_-1.tsv`.

The model outputs correspond to pronunciations and these are split on space by
the decoder using `--split_output_on_space` flag to match the corresponding
operation during the data preparation stage.

The `-1` in file names above refers to the checkpoint limit, set to `-1` by
default if `--ckpt_limit` flag is unspecified in the decoder. This basically
means that the latest model checkpoint was evaluated.

### Shared Task: Language Group as a Single Model Submission

In this configuration we train a single model for the entire language group.
This is accomplished by setting the `--lang` flag to the (case-insensitive)
value `all` instead of a language name when generating the training data
using the `create_neighborhood.py` tool. Example (for `beidazihui` language group):

```shell
  python neighbors/data/create_neighborhood.py \
    --task_data_dir ${DATA_DIR}/data-surprise \
    --output_dir ${OUTPUT_DATA_DIR} \
    --language_group beidazihui --lang all \
    --pairwise_algo lingpy --random_target_algo markov \
    --max_rand_len 10 --num_duplicates 100 --logtostderr
```

Upon completion the above command will output three parameters which are
important for configuring the neighborhood embeddings: the number of
neighbors, the maximum language name length and the maximum pronunciation
length. We are adding 1 to the pronunciation and language name lengths.
For the surprise data these are as follows:

| Group     | Max Neighbors | Max Name Length | Max Pron Length |
| ----------- | ----------- | --------------- | --------------- |
| `bantubvd` | 9 | 3 | 11 |
| `beidazihui` | 14 | 16 | 6 |
| `birchallchapacuran` | 9 | 9 | 15 |
| `bodtkhobwa` | 7 | 9 | 6 |
| `bremerberta` | 3 | 14 | 12 |
| `deepadungpalaung` | 15 | 14 | 8 |
| `hillburmish` | 8 | 16 | 7 |
| `kesslersignificance` | 4 | 9 | 11 |
| `luangthongkumkaren` | 7 | 12 | 8 |
| `wangbai` | 9 | 10 | 6 |

The above parameters should be supplied to both the training tool `trainer.py`
and the decoder tool `decoder.py` via the following flags: `--max_neighbors`,
`--max_spelling_len` and `--max_pronunciation_len`. Example (for `beidazihui`
language group):

```shell
  # Training.
  python neighbors/model/trainer.py \
    --run_locally=cpu \
    --model=feature_neighborhood_model_config.TransformerWithNeighborsTiny \
    --input_symbols ${OUTPUT_DATA_DIR}/beidazihui.syms \
    --output_symbols ${OUTPUT_DATA_DIR}/beidazihui.syms \
    --feature_neighborhood_train_path tfrecord:${OUTPUT_DATA_DIR}/beidazihui_train.tfrecords \
    --feature_neighborhood_dev_path tfrecord:${OUTPUT_DATA_DIR}/beidazihui_test.tfrecords \
    --feature_neighborhood_test_path tfrecord:${OUTPUT_DATA_DIR}/beidazihui_test.tfrecords \
    --learning_rate=0.001 \
    --max_neighbors=14 \
    --max_pronunciation_len=6 \
    --max_spelling_len=16 \
    --logdir ${MODEL_TRAIN_DIR}

  # Decoding.
  python neighbors/model/decoder.py \
    --model=TransformerWithNeighborsTiny \
    --feature_neighborhood_test_path=tfrecord:${OUTPUT_DATA_DIR}/beidazihui_test.tfrecords \
    --input_symbols ${OUTPUT_DATA_DIR}/beidazihui.syms \
    --output_symbols ${OUTPUT_DATA_DIR}/beidazihui.syms \
    --ckpt ${MODEL_TRAIN_DIR}/train \
    --decode_dir ${RESULTS_DIR} \
    --split_output_on_space \
    --max_neighbors=14 \
    --max_pronunciation_len=6 \
    --max_spelling_len=16 \
    --batch_size 1
```

The results file in Shared Task TSV format is  `${RESULTS_DIR}/results_-1.tsv`.

## Cognate Inpainting Model

The model is inspired by image infill/inpainting
[work](https://arxiv.org/pdf/1804.07723.pdf) by NVidia. There, random masks are
applied to images to blank out certain pixels, and a convolutional network is
used to restore those pixels given the surrounding context.

### Training

For training, non-default hyperparameters (`embedding_dim`, `kernel_width`,
`filters`, `dropout`, `nonlinearity`, `sfactor`) must be specified as flags.

```shell
  DATA_DIR=~/projects/ST2022
  MODEL_TRAIN_DIR=/tmp/logdir
  python inpaint/cognate_inpaint.py \
    --embedding_dim 16 \
    --kernel_width 1 \
    --filters 32 \
    --dropout 0.6 \
    --nonlinearity tanh \
    --sfactor inputs \
    --data_dir ${DATA_DIR} \
    --train_file train-0.10.tsv \
    --dev_file dev-0.10.tsv \
    --dev_solutions_file dev_solutions-0.10.tsv \
    --checkpoint_dir ${MODEL_TRAIN_DIR}
```

### Inference

The checkpoint dir should contain an `hparams.json` file and `vocab.txt` file,
in addition to a valid checkpoint.

```shell
  RESULTS_FILE=/tmp/results-0.10.tsv
  python inpaint/cognate_inpaint.py \
    --data_dir ${DATA_DIR} \
    --test_file test-0.10.tsv \
    --preds_file model_result-0.10.tsv \
    --checkpoint_dir ${MODEL_TRAIN_DIR} \
    --decode
```

### Ensemble Multiple Results Files

An arbitrary set of identically-shaped results files can be ensembled via
majority vote. Filepaths can be provided as a comma-separated list.

```shell
  python inpaint/ensemble_results.py \
    --input_results_tsvs=/path/to/result-1.tsv,path/to/result-2.tsv \
    --output_results_tsv=/path/to/ensemble.tsv
```
## Citation

If you use this software in a publication, please cite the accompanying
[paper](https://aclanthology.org/2022.sigtyp-1.9/) presented at
[SIGTYP 2022](https://sigtyp.github.io/workshop.html):

```bibtex
@inproceedings{kirov-etal-2022-mockingbird,
    title = "Mockingbird at the {SIGTYP} 2022 Shared Task: Two Types of Models for the Prediction of Cognate Reflexes",
    author = "Kirov, Christo  and
      Sproat, Richard  and
      Gutkin, Alexander",
    booktitle = "Proceedings of the 4th Workshop on Research in Computational Linguistic Typology and Multilingual NLP",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sigtyp-1.9",
    doi = "10.18653/v1/2022.sigtyp-1.9",
    pages = "70--79"
}
```
