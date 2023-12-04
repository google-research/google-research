This repository contains code for attribution reporting API optimization for the
paper

> Hidayet Aksu, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi,
> Adam Sealfon, Avinash Varadarajan.\
> [*Summary Reports Optimization in the Privacy Sandbox Attribution Reporting
> API*.](https://arxiv.org/abs/2311.13586)

**Note**: This is not an officially supported Google product.

## Abstract

## Getting Started

### Requirements

This codebase uses [Numpy](https://numpy.org/), [Scipy](https://scipy.org/)
and [Pandas](https://pandas.pydata.org/).

You can install the dependencies using:
```
pip install -r requirements.txt
```

### Run ARA Summary Report Optimization on a dataset

Given training and test Pandas dataframes `df_train` and `df_test`, a list of
strings `df_slice_columns` specifying the dataframe columns used to define the
slices, a list of strings `df_value_columns` specifying the dataframe columns
used to define the variables being queried, and a string `df_count_column`
specifying the dataframe column containing the count variable, use the following
commands to run:

```python
dataset_rmsre_metrics = util.rmsre_tau_error_metrics(df_train, df_value_columns
                                                     + [df_count_column])
result = run_experiment.run_experiment(df_train, df_test, df_slice_columns,
                                       df_value_columns, df_count_column,
                                       dataset_rmsre_metrics)
```

To generate/regenerate synthetic datasets run

```
python -m ara_optimization.synthetic_data_exports
```

command which generates data inside ./synthetic-datasets folder.
