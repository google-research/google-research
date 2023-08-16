# A Unified Model for Graph Structure Learning (UGSL)

This is the official repository for our paper, `UGSL: A Unified Framework for Benchmarking Graph Structure Learning`.
The repo contains a unified framework with different components in it for graph structure learning models.

## Usage

### Installation

Download the code (e.g., via git clone or direct download).

You can first install requirements as:

```sh
python -m pip install requirements.txt
```


### Your first run

First, let us define the environment variable `$BIN` to be our executable

```sh
BIN=python main.py
```

Of course, you may set this in accordance with your setup (e.g., if you use `python3`, `anaconda`, or have additional custom flags).

Then, run your first experiment, with:
```sh
$BIN
```

### Custom models and datasets

The file `config.py` contains the default configuration: trains a GCN encoder on *cora*, with graph structure being learned via an *MLP edge-scorer* and a *KNN sparsifier*.

You may override, e.g., the dataset with:

```sh
$BIN --config.dataset.name amazon_photos
```

You may also provide another python file that contains many different modifications. For instance, copy `config.py` onto (e.g.) `path/to/new_config.py` and modify it, as you wish. Then you can run like:

```sh
$BIN --config path/to/new/config.py
```

You may still override any config variable, e.g., to use the fully-parameterized edge-scorer:

```sh
$BIN --config path/to/new/config.py --config.model.edge_scorer_cfg.name=fp
```
Regardless of the `model.edge_scorer_cfg` utilized in `path/to/new/config.py`, the value passed to the command-line takes precedence.

## Contact us

For questions or comments about the implementation, please contact baharef@google.com.

## Disclaimer

This is not an official Google product.
