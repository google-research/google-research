# Optimizing Generalized Rate Metrics with Three Players

Example code for running experiments for algorithms in:
Harikrishna Narasimhan, Andrew Cotter, Maya Gupta, "Optimizing Generalized Rate Metrics with Three Players", NeurIPS 2019. https://arxiv.org/pdf/1909.02939

## Installation

Run the following:
```shell
virtualenv -p python3 .
source ./bin/activate

pip3 install -r generalized_rates/requirements.txt
```

## Running F-measure Optimization on COMPAS dataset

Download the COMPAS dataset from:
https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv
and save it in the `generalized_rates/datasets` folder.

Run the following:
```shell
DATA_DIR=./generalized_rates/datasets
python -m generalized_rates.datasets.load_compas\
  --data_file=$DATA_DIR/compas-scores-two-years.csv\
  --output_directory=$DATA_DIR/
python -m generalized_rates.fmeasure_optimization.experiments\
  --data_file=$DATA_DIR/COMPAS.npy
```

## Running KLD Optimization on Adult dataset

Download the Adult train and test data files can be downloaded from:
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
and save them in the `generalized_rates/datasets` folder.

Run the following:
```shell
DATA_DIR=./generalized_rates/datasets
python -m generalized_rates.datasets.load_adult\
  --train_file=$DATA_DIR/adult.data\
  --test_file=$DATA_DIR/adult.test\
  --output_directory=$DATA_DIR/
python -m generalized_rates.kld_optimization.experiments\
  --data_file=$DATA_DIR/Adult.npy
```
