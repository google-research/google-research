
# Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
This repository contains a TensorFlow implementation of "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks" by Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh (accepted as ORAL presentation in ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) 2019).

Paper link: https://arxiv.org/pdf/1905.07953.pdf

## Requirements

* install clustering toolkit: metis and its Python interface.

  download and install metis: http://glaros.dtc.umn.edu/gkhome/metis/metis/download

  METIS - Serial Graph Partitioning and Fill-reducing Matrix Ordering ([official website](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview))

```
1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2) cd metis-5.1.0
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so
```

* install required Python packages

```
 pip install -r requirements.txt
```

quick test to see whether you install metis correctly:

```
>>> import networkx as nx
>>> import metis
>>> G = metis.example_networkx()
>>> (edgecuts, parts) = metis.part_graph(G, 3)
```

* We follow [GraphSAGE](https://github.com/williamleif/GraphSAGE#input-format)'s input format and its code for pre-processing the data.

* This repository includes scripts for reproducing our experimental results on PPI and Reddit. Both datasets can be downloaded from this [website](http://snap.stanford.edu/graphsage/).

## Run Experiments.

* After metis and networkx are set up, and datasets are ready, we can try the scripts.

* We assume data files are stored under './data/{data-name}/' directory.

  For example, the path of PPI data files should be: data/ppi/ppi-{G.json, feats.npy, class_map.json, id_map.json}

* For PPI data, you may run the following scripts to reproduce results in our paper

```
./run_ppi.sh
```

  For reference, with a V100 GPU, running time per epoch on PPI is about 1 second.

```
The test F1 score will be around 0.9935 depending on different initialization.

```

* For reddit data (need change the data_prefix path in .sh to point to the data):

```
./run_reddit.sh
```
Below shows a table of state-of-the-art performance from recent papers.

|               | PPI         | Reddit    |
| ------------- |:-----------:| ---------:|
| [FastGCN](https://arxiv.org/abs/1801.10247) ([code](https://github.com/matenure/FastGCN))           | N/A         | 93.7      |
| [GraphSAGE](https://arxiv.org/abs/1706.02216) ([code](https://github.com/williamleif/GraphSAGE))    | 61.2        | 95.4      |
| [VR-GCN](https://arxiv.org/abs/1710.10568) ([code](https://github.com/thu-ml/stochastic_gcn))       | 97.8        | 96.3      |
| [GAT](https://arxiv.org/abs/1710.10903) ([code](https://github.com/PetarV-/GAT))                    | 97.3        | N/A       |
| [GaAN](https://arxiv.org/abs/1803.07294)                                                     | 98.71       | 96.36     |
| [GeniePath](https://arxiv.org/abs/1802.00910)                                                | 98.5        | N/A       |
| Cluster-GCN                                                  | **99.36**   | **96.60** |

If you use any of the materials, please cite the following paper.

```
@inproceedings{clustergcn,
  title = {Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks},
  author = { Wei-Lin Chiang and Xuanqing Liu and Si Si and Yang Li and Samy Bengio and Cho-Jui Hsieh},
  booktitle = {ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year = {2019},
  url = {https://arxiv.org/pdf/1905.07953.pdf},
}

