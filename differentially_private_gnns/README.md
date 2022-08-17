# Node-Level Differentially Private Graph Neural Networks

A JAX implementation of 'Node-Level Differentially Private Graph
Neural Networks', presented at PAIR2Struct, ICLR 2022.

### Instructions

Clone the repository:

```shell
git clone https://github.com/google-research/google-research.git
cd google-research/differentially_private_gnns
```

Create and activate a virtual environment:

```shell
python -m venv .venv && source .venv/bin/activate
```

Install dependencies with:

```shell
pip install --upgrade pip && pip install -r requirements.txt
```

Download a dataset (ogbn-arxiv shown below) with our script:

```shell
python download_datasets.py --dataset_name=ogbn-arxiv
```

Start training with a configuration defined
under `configs/`:

```shell
python main.py --workdir=./tmp --config=configs/dpgcn.py
```

#### Changing Hyperparameters

Since the configuration is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags),
you can override hyperparameters. For example, to change the number of training
steps, the batch size and the dataset:

```shell
python main.py --workdir=./tmp --config=configs/dpgcn.py \
--config.num_training_steps=10 --config.batch_size=50 \
--config.dataset=reddit-disjoint
```

If you just want to test training without any dataset downloads,
you can also run the end-to-end training test on the dummy dataset:

```shell
python train_test.py
```

For more extensive changes, you can directly edit the configuration files,
and even add your own.

#### Changing Dataset Paths

By default, the datasets are downloaded to `datasets/` in the current working
directory. You can change this by passing `dataset_path` to the download script:

```shell
python download_datasets.py --dataset_name=ogbn-arxiv \
--dataset_path=${DATASET_PATH}
```

and then updating the path in the config:

```shell
python main.py --workdir=./tmp --config=configs/dpgcn.py \
--config.dataset_path=${DATASET_PATH}
```

### Notes

This is a simpler and faster JAX implementation
that differs from the TensorFlow implementation
used to obtain results in the paper.
The main constraint is because XLA requires
fixed-size arrays to represent subgraphs.

### Citation

Please cite our paper if you use this code!

```text
@inproceedings{
daigavane2022nodelevel,
title={Node-Level Differentially Private Graph Neural Networks},
author={Ameya Daigavane and Gagan Madan and Aditya Sinha and Abhradeep Guha Thakurta and Gaurav Aggarwal and Prateek Jain},
booktitle={ICLR 2022 Workshop on PAIR{\textasciicircum}2Struct: Privacy, Accountability, Interpretability, Robustness, Reasoning on Structured Data},
year={2022},
url={https://openreview.net/forum?id=BCfgOLx3gb9}
}
```

