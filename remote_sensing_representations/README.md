# In-Domain Representation Learning for Remote Sensing

Repository related to: "In-Domain Representation Learning for Remote Sensing"
by Maxim Neumann, André Susano Pinto, Xiaohua Zhai and Neil Houlsby.

Paper: [arxiv.org/abs/1911.06721](https://arxiv.org/abs/1911.06721).

Pre-trained representations are available in [TensorFlow Hub](https://tfhub.dev/google/collections/remote_sensing/1)

Datasets are available in
[TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog):

* https://www.tensorflow.org/datasets/catalog/bigearthnet
* https://www.tensorflow.org/datasets/catalog/eurosat
* https://www.tensorflow.org/datasets/catalog/resisc45
* https://www.tensorflow.org/datasets/catalog/so2sat
* https://www.tensorflow.org/datasets/catalog/uc_merced


## Dataset splits
The python notebook [rsr_datasets.ipynb](rsr_datasets.ipynb) shows how to
generate the exact splits used in the paper directly from the TensorFlow
datasets.

For convenience, the splits (denoting filenames or sample IDs) can also be found in:

### BigEarthNet
- https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt
- https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt
- https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt

### EuroSAT
- https://storage.googleapis.com/remote_sensing_representations/eurosat-train.txt
- https://storage.googleapis.com/remote_sensing_representations/eurosat-val.txt
- https://storage.googleapis.com/remote_sensing_representations/eurosat-test.txt

### RESISC-45
- https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt
- https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt
- https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt

### So2Sat
- https://storage.googleapis.com/remote_sensing_representations/so2sat-train.txt
- https://storage.googleapis.com/remote_sensing_representations/so2sat-val.txt
- https://storage.googleapis.com/remote_sensing_representations/so2sat-test.txt

### UC Merced
- https://storage.googleapis.com/remote_sensing_representations/uc_merced-train.txt
- https://storage.googleapis.com/remote_sensing_representations/uc_merced-val.txt
- https://storage.googleapis.com/remote_sensing_representations/uc_merced-test.txt

## Citation
```
@inproceedings{rsr:iclr2020,
  author    = {M. Neumann and A. S. Pinto and X. Zhai and N. Houlsby},
  title     = {In-domain representation learning for remote sensing},
  booktitle = {AI for Earth Sciences Workshop at International Conference on Learning Representations (ICLR)},
  month     = apr,
  year      = {2020},
  pages     = {1-20},
  url       = {https://arxiv.org/abs/1911.06721},
}
```

```
@inproceedings{rsr:igarss2020,
  author    = {M. Neumann and A. S. Pinto and X. Zhai and N. Houlsby},
  title     = {Training general representations for remote sensing using in-domain knowledge},
  booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  publisher = {{IEEE}},
  month     = sep,
  year      = {2020},
  pages     = {1-4},
}
```
