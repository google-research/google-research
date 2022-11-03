# Codebase for "Invariant Structure Learning for Better Generalization and Causal Explainability (ISL)"

This directory contains the implementation of ISL for the following use cases

- Building environments for both synthetic data and real-world data
- Running ISL on supervised setting
- Running ISL on self-supervised setting

The structure of the content:
- "data" folder contains processed data for Boston Housing, Insurance and synthetic datasets.
- "data_gen_real.py" builds environments for real-world datasets.
- "data_gen_syn.py" builds environments for synthetic datasets.
- "ISL_modula.py" contains source code of the implementation of the ISL module and core functions.
- "utils.py" contains source code for data processing, save/load, visualize, evaluation metrics.
- "main_ISL_supervised.py" runs ISL in supervised learning setting.
- "main_ISL_selfsupervise" runs ISL in self-supervised learning setting.


# Getting started

### Dependencies
```shell
pip install -r requirements.txt
```

### Data preprocess or generation

- If you want to generate synthesic data and build environments, follow the example:
```shell
python data_gen_syn.py --3 --1 --100 --3
```
to create a graph with 2 X variable and 1 S variable in 3 environments each containing 100 samples.


For real data, we use the following datasets, that you should download directly:

Boston Housing (http://lib.stat.cmu.edu/datasets/boston)

Insurance (https://link.springer.com/article/10.1023/A:1007421730016)

Sachs (https://www.science.org/doi/full/10.1126/science.1105809)


- To build environments for real data:
```shell
python data_gen_real.py --BH --standard --3
```
to load the standardized BH dataset clustered into 3 environments.

the generated and processed data will be saved in "data" folder

### ISL Supervised Learning

- If you want to conduct ISL on supervised tasks, run "main_isl_supervised.py",
one example

```shell
python main_isl_supervised.py --dataset_name 'Sachs' --normalization standard \
--lambda2_Y_fc1 0 --lambda2_Y_fc2 0.01 --lambda1_Y 0.0001 --hidden 10 \
--Y_hidden 10 --Y_hidden_layer 5 --NOTEAR_hidden_layer 1 --beta 1 --y_index 7
```
It will utilize the functions in "isl_module.py" and "utils.py"

### ISL Self-Supervised Learning

- The first step is iteratively set each variable as Y and call "main_ISL.py".
- We need to manually aggregate the proposal in to a DAG and document them in "main_self_supervise.py".
- The last step is to conduct ISL on Self-Supervised tasks, use "main_self_supervised.py",
as an example:

```shell
python main_ISL_selfsupervised.py --dataset_name 'Sachs'
```


### Adapting to a new tabular dataset

If you have your own tabular dataset, you can use our code with the following steps:

- First, build different environments using "data_gen_real.py" and save them by creating a "./data" folder.
For this step, you can put the tabular date into a .csv form, where each row is a sample and column number represents the number of variables.

The "data_gen_real.py" will take the original tabular data as input and generate dataset in different environments.

- If your task is supervised task (interested in Y prediction and Y-related DAG), follow the section "ISL Supervised Learning" above.

Specifically, "main_ISL_supervised.py" takes the environment data as input and conduct ISL, the output would be a Y classifier and Y-related DAG.

- If your task is self-supervised task (interested in dataset DAG), follow section "ISL Self-Supervised Learning" above.

The first step is iteratively set each variable as Y and call "main_ISL.py".
We need to manually aggregate the proposal into a DAG and refer them in "main_selfsupervised.py". We select the causal parents summarized from ISL for each variable and fill a initial DAG into main_selfsupervised.py.


The last step is to conduct ISL on Self-Supervised tasks, using "main_self_supervise.py".
