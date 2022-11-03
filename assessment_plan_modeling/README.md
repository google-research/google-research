# Assessment and Plan Modeling

Code for the paper *"Structured Understanding of Assessment and Plans in
Clinician Documentation"* by Stupp et al. Available as a preprint on medrxiv:
https://www.medrxiv.org/content/10.1101/2022.04.13.22273438v1

Citation for the paper:

``` {bibtex}
Citation TBA
```

The annotations dataset accompanying the paper can be found on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6413405.svg)](https://doi.org/10.5281/zenodo.6413405)

# Instructions for running

The full model training process can be run using the `run.sh` script, which also
creates a suitable virtual environment.

To run individual steps, consult the instructions below.

## Prerequisites

Python3 is required, code was tested on 3.9.9 but should work on 3.7+. Create
and active a virtual environment:

``` {bash}
python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
```

Tests are executable using the test specific module, for example:

``` {bash}
python3 -m assessment_plan_modeling.ap_parsing.ap_parsing_task_test
```

## Data Generation

To generate TensorFlow examples for model training, run the beam pipeline
presented in `data_gen_main.py`. Input notes are expected to be in the MIMIC-III
note_event csv format. Annotations (`all_model_ratings.csv`) can be downloaded
from the aforementioned dataset.

Unfortunately as the vocabulary contains MIMIC-III data it is up to the user to
generate an appropriate vocabulary. The file consists of unquoted, escaped
strings, with each token on a new line. An example vocabulary is found in the
repository head `sample_vocab.txt`.

``` {bash}
DATA_DIR="path/to/data"
python assessment_plan_modeling/ap_parsing/data_gen_main.py \
  --input_note_events="${DATA_DIR}/notes.csv" \
  --input_ratings="${DATA_DIR}/all_model_ratings.csv" \
  --output_path="${DATA_DIR}/ap_parsing_tf_examples/$(date +%Y%m%d)" \
  --vocab_file="${DATA_DIR}/sample_vocab.txt" \
  --section_markers="assessment_plan_modeling/note_sectioning/mimic_note_section_markers.json" \
  --n_downsample=100 \
  --max_seq_length=2048
```

## Model Training

Model training is configurable by yaml as described in the TensorFlow model
garden
[repository](https://github.com/tensorflow/models/tree/master/official/nlp).

### Test Run

Using dummy data, should complete in several seconds on CPU.

``` {bash}
MODEL_DIR="/tmp/test_model_$(date +%Y%m%d-%H%M)"
WORK_DIR="$PWD/assessment_plan_modeling/ap_parsing"
EXP_TYPE="ap_parsing"
CONFIG_DIR="${WORK_DIR}/configs"
python ${WORK_DIR}/train.py \
  --experiment=${EXP_TYPE} \
  --config_file="${CONFIG_DIR}/local_example.yaml" \
  --mode=train \
  --model_dir=${MODEL_DIR} \
  --alsologtostderr
```

### Full Run

Running on a single V100 GPU, this should take approximately 2 hours.

``` {bash}
MODEL_DIR="/tmp/model_$(date +%Y%m%d-%H%M)"
WORK_DIR="$PWD/assessment_plan_modeling/ap_parsing"
EXP_TYPE="ap_parsing"
CONFIG_DIR="${WORK_DIR}/configs"
DATA_DIR="/path/to/tfrecords"

TRAIN_DATA="${DATA_DIR}/train_rated_nonaugmented.tfrecord*"
VAL_DATA="${DATA_DIR}/val_set.tfrecord*"
PARAMS_OVERRIDE="task.use_crf=true"
PARAMS_OVERRIDE="${PARAMS_OVERRIDE},task.train_data.input_path='${TRAIN_DATA}'"
PARAMS_OVERRIDE="${PARAMS_OVERRIDE},task.validation_data.input_path='${VAL_DATA}'"
PARAMS_OVERRIDE="${PARAMS_OVERRIDE},trainer.train_steps=5000"

python ${WORK_DIR}/train.py \
  --experiment=${EXP_TYPE} \
  --config_file="${CONFIG_DIR}/base_ap_parsing_model_config.yaml" \
  --config_file="${CONFIG_DIR}/base_ap_parsing_task_config.yaml" \
  --params_override=${PARAMS_OVERRIDE} \
  --mode=train_and_eval \
  --model_dir=${MODEL_DIR} \
  --alsologtostderr
```

### Inference and Evaluation

Inference and evaluation libraries are supplied as is.
