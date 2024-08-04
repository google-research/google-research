This repository implements a benchmark and solutions for the challenge of
transferring an inductive temporal link prediction model from one graph to
test graph that has no node intersection with the initial (training) graph.

## Env setup

To run experiments in this repo, you'll need to set up a conda directory. There
is currently no other known option as one of the libraries (graph-tool) is not
pip-installable. First, run these commands to set up the env:

```
ENV_NAME=yourenvname
conda create --name $ENV_NAME -c conda-forge graph-tool
conda activate ${ENV_NAME}
```

At this point, to install the other Python modules, you'll need to locate the
Conda environment's version of Python (with the highest version number) that is
able to load `graph-tool`. On Linux systems this is usually a `python3.XX` binary
in a directory like

```
/usr/local/home/${USER}/anaconda3/envs/${ENV_NAME}/bin
```

Unfortunately due to some installation oddities arising from the interaction
between Conda and `graph-tool`, simply running `python` with the environment
activated will not start the correct Python version. Find `python3.XX` in the
folder above (or the analogous folder on your system). Then set the alias

```
alias python3_env="/usr/local/home/${USER}/anaconda3/envs/${ENV_NAME}/bin/python3.XX"
```

Using this alias, install torch (you may be able to up the version if PyG keeps up):

```
python3_env -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest of the packages (checking the torch version if necessary):

```
TORCH_V=2.3.0+cu118
python3_env -m pip install torch-geometric
python3_env -m pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_V}.html
python3_env -m pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_V}.html
python3_env -m pip install tensorflow
python3_env -m pip install pandas
python3_env -m pip install scikit-learn
python3_env -m pip install tqdm
python3_env -m pip install apache-beam
```

## End-to-end with `tgbl-wiki`
This section walks through an end-to-end run with the `tgbl-wiki` dataset from
the [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com/).


### Dataset prep
We start from the dataset as provided by TGB. Create a datasets directory as
follows:

```
PROJECT_ROOT=/where/you/will/keep/datasets/and/experiments
mkdir -p ${PROJECT_ROOT}/datasets/tgbl_wiki
```

Put the `tgbl-wiki_edgelist_v2.csv` file, linked to at the
[TGB dataset resources page](https://github.com/shenyangHuang/TGB/blob/main/tgb/utils/info.py),
in the directory above. Then, from our repository directory, run the following
command to prepare the dataset. This job runs Louvain community detection on
the full graph, once the graph has been loaded.

```
python3_env tgbl_wiki_dataprep.py --root_dir=${PROJECT_ROOT}
```

Next, to enable transfer learning experiments, run the job below, which aggregates
communities into separate train and test graphs for transfer learning.

```
python3_env tgbl_wiki_subgraph_prep.py --root_dir=${PROJECT_ROOT}
```

Note: for the `tgbl_flight` dataset, skip the above step. For transfer learning
experiments on airports, we use continent codes to divide nodes instead of
community detection. Run the following to prepare the continent-wise datasets:

```
for continent in OC EU AN SA AS AF;
do
  python3_env tgbl_flight_negatives.py --root_dir=${PROJECT_ROOT} --continent=$continent
done
```

Next we precompute the per-batch subgraphs:

```
for SPLIT in train val test;
do
  python3_env generate_subgraphs.py \
    --root_dir=${PROJECT_ROOT} \
    --community=cc-subgraph \
    --split=${SPLIT} \
    --data=tgbl_wiki
done
```

Then we compute per-batch structural features:

```
python3_env generate_structural_features_pipeline.py \
  --data='tgbl_wiki;cc-subgraph' \
  --root_dir=${PROJECT_ROOT} \
  --bs=200 \
  --pos_enc_dim=4 \
  --aggregation_window_frac=0.01
"""
```

### Train / test
The training scripts depend on a few protobufs. Install `protoc`:

```
PROTOC_V=3.20.3
PROTOC_ZIP=protoc-${PROTOC_V}-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_V}/${PROTOC_ZIP}
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

Then run the following while in the folder containing `train.py`:

```
protoc models/model_config.proto --python_out=./
```

Now we can kick off the train job:

```
python3_env train.py \
  --data=tgbl_wiki \
  --num_epoch=5 \
  --root_dir=${PROJECT_ROOT} \
  --output_subdir=readme_experiment \
  --model_name=edgebank \
  --train_group=cc-subgraph \
  --val_group=cc-subgraph
```

After that completes, run test:

```
python3_env test.py \
  --data=tgbl_wiki \
  --root_dir=${PROJECT_ROOT} \
  --output_subdir=readme_experiment \
  --model_name=edgebank \
  --train_group=cc-subgraph \
  --val_group=cc-subgraph \
  --test_group=cc-subgraph
```

