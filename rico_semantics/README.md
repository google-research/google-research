# Semantic Annotations for the RICO dataset: Towards Better Semantic Understanding of Mobile Interfaces

## Overview

The respository contains the data processing and modeling code related to the paper: `Towards Better Semantic Understanding of Mobile Interfaces` ([paper](http://ranjithakumar.net/resources/rico.pdf)). The code includes data processing code to combine the annotations with the raw RICO data to create the training and eval data for the models, code to run training and evaluation of models like DETR.

## Important Links

*   [Paper - Towards Better Semantic Understanding of Mobile Interfaces](https://arxiv.org/abs/2210.02663)
*   [Dataset location](https://github.com/google-research-datasets/rico-semantics)
*   [Original RICO dataset](https://interactionmining.org/rico) and [paper](http://ranjithakumar.net/resources/rico.pdf).

## Usage

### Data Processing
The data processing code helps to combine the labels data into `.tfrecord` format which can be used for training and evaluating the models.

First, download the raw RICO data to a directory named `raw_data`.

Second, download the annotations data to a diretory named `annotations`.

Then, `cd` to the `google_reserach` directory and install the requirements by running:

```
pip install -r rico_semantics/requirements.txt
```

Then you can run the following command to convert the data into `.tfrecord` format, which is used for training the models. The parameter `task_name` can take one of `[icon_shape, icon_semantics, label_association]` as the valid values.

```
python -m rico_semantics.convert_raw_data_to_tfrecords --data_path=raw_data/ --task_name=icon_semantics --annotations_dir=annotations/ --output_dir=/tmp/tfrecords/
```

## Cite Us
If you use the dataset or the data processing and modeling code in your work, please cite our work as follows:

```shell
@article{https://doi.org/10.48550/arxiv.2210.02663,
  author    = {Srinivas Sunkara and
               Maria Wang and
               Lijuan Liu and
               Gilles Baechler and
               Yu-Chung Hsiao and
               Jindong Chen and
               Abhanshu Sharma and
               James Stout},
  title     = {Towards Better Semantic Understanding of Mobile Interfaces},
  journal   = {CoRR},
  volume    = {abs/2210.02663},
  year      = {2022},
  url       = {https://arxiv.org/abs/2210.02663},
  eprinttype = {arXiv},
  eprint    = {2201.12409},
  timestamp = {Thu, 06 Oct 2022 15:00:01 +0100},
}
```
