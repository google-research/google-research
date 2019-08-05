
# Cluster Graph convolutional network (Cluster-GCN)
This repository contains the code behind "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks" by Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh (accepted as ORAL presentation in ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) 2019).

Arxiv link: https://arxiv.org/pdf/1905.07953.pdf

## Requirements

* install clustering toolkit: metis and its Python interface.

download and install metis: http://glaros.dtc.umn.edu/gkhome/metis/metis/download

```
1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download
2) cd to the metis folder
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so
```

* install requirements.txt

```
 pip install -r requirements.txt
```

NOTE that we are using tensorflow 1.14.0 version here. For prerequirement for installing tensorflow 1.14.0, please check https://www.tensorflow.org/install/source#tested_build_configurations. We recommend to use gcloud instance.

quick test whether you install metis correctly:

```
>>> import networkx as nx
>>> import metis
>>> G = metis.example_networkx()
>>> (edgecuts, parts) = metis.part_graph(G, 3)
```

* This project includes some codes from https://github.com/williamleif/GraphSAGE to process the data, and make sure your data is in graphsage format.

* This code includes two scripts for testing cluster GCN method. One is on PPI data and another on Reddit data. Please download both datasets before running the scripts. The two datasets can be downloaded from http://snap.stanford.edu/graphsage/.

## Run Experiments.

* After metis and networkx are set up, and datasets are ready, we can try the scripts.

* put data under the './data' path.

For example the ppi data folder should look like: data/ppi/ppi-{G.json, feats.npy, class_map.json, id_map.json}

* For ppi data (need change the data_prefix path in .sh to point to the data):

```
./run_ppi.sh
```

If testing on V100 on google cloud machine, per epoch training time will be around ~1 second.

```
The test F1 score will be around 0.9935 depending on different initialization.

```

* For reddit data (need change the data_prefix path in .sh to point to the data):

```
./run_reddit.sh
```
If you use any of the material here please cite the following paper:

```
@inproceedings{clustergcn,
  title = {Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks},
  author = { Wei-Lin Chiang and Xuanqing Liu and Si Si and Yang Li and Samy Bengio and Cho-Jui Hsieh},
  booktitle = {ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year = {2019},
  url = {https://arxiv.org/pdf/1905.07953.pdf},
}

