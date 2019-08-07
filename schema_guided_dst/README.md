# Evaluation Script for DSTC8 Schema Guided Dialogue State Tracking

**Contact -** schema-guided-dst@google.com

## Required packges
1. fuzzywuzzy
2. numpy
3. tensorflow

## Dataset:
[dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

## How to run:

### Calculate DSTC8 Metrics
Evaluation is done using `evaluate.py` which calculates the values of different
metrics defined in `metrics.py` by comparing model outputs with ground truth.
The script `evaluate.py` requires that all model predictions should be saved in
one or more json files contained in a single directory (passed as flag
`prediction_dir`). The json files must have same format as the ground truth data
provided in the challenge repository. This script can be run using the following
command:


```shell
python -m evaluate.py \
--alsologtostderr \
--prediction_dir=<directory containing model outputs> \
--dstc8_data_dir=<dstc data directory downloaded from challenge repo> \
--eval_set=<dataset, one of 'train', 'dev' or 'test'> \
--output_metric_file=<path to json file where evaluation report will be saved>
```
