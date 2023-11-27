This repository contains code for attribution reporting API optimization for the
paper

> Hidayet Aksu, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi,
> Adam Sealfon, Avinash Varadarajan.\
> *Summary Reports Optimization in the Privacy Sandbox Attribution Reporting
> API*.

**Note**: This is not an officially supported Google product.

## Abstract

## Getting Started

### Requirements

This codebase uses [Numpy](https://numpy.org/), [Scipy](https://scipy.org/)
and [Pandas](https://pandas.pydata.org/).

### Run ARA Summary Report Optimization on a dataset

Given training and test Pandas dataframes `df_train` and `df_test`, a list of
slice columns `df_slice_columns`, a list of query columns `df_value_columns`,
and a count column `df_count_column`, use the following command to run:

```
python
dataset_rmsre_metrics = util.rmsre_tau_error_metrics(df_train, df_value_columns
                                                     + [df_count_column])
result = run_experiment.run_experiment(df_train, df_test, df_slice_columns,
                                       df_value_columns, df_count_column,
                                       dataset_rmsre_metrics)
```

To generate/regenerate synthetic datasets run
```
python -m privacy.ara.synthetic_data_exports.py
```
command which generates data inside ./synthetic-datasets folder.
