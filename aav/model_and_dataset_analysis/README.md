# Runnable code to reproduce results presented

## Folder contents

Jupyter notebooks that reproduce the paper results:
* Figure 2 notebook
* Figure 3 notebook (also includes SI Tables)
* Figure 4 notebook

Data for reproducing results:
* AAV2 capsid sequences (allseqs_20191230.csv.zip)
  - Capsid packaging assay results for 297k capsid sequences
  - Model training and validation datasets
  - Baseline additive and random datasets
  - Model-designed sequences
  - Model-selected sequences
* Subsampled model-designed sequence partitions (subsampled.tar.gz)
  - Used to normalize sequence subset sizes prior to clustering
* Clustering of all (viable + non-viable) model-designed sequences (clusters.tar.gz)
* Clustering of viable-only model-designed sequences (viable_clusters.tar.gz)


## Setup Python environment

### Install Anaconda distribution of Python

https://www.anaconda.com/products/individual
Python 2.7 64-Bit Graphical Installer (637 MB)

Note: instructions have been tested on Mac OS 10.14; Anaconda installers are also available for Windows and Linux.

### Install Python dependencies

$ conda install --file requirements.txt


## Run notebooks

### Start Jupyter notebook server in code directory

$ cd path/to/this/folder
$ jupyter notebook .

Navigate your web browser to the URL printed in the console by Jupyter (typically http://localhost:8888).
