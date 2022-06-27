# Federated Vision Datasets

This repo contains the data splits from the following paper.

> Federated Visual Classification with Real-World Data Distribution<br>
Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown<br>
ECCV 2020<br>
https://arxiv.org/abs/2003.08082

## Datasets

Dataset               | # users     | # classes | # examples | Download split files          | For TensorFlow Federated   |
--------------------- | ----------: | --------: | ---------: | ----------------------------- |----------------------------|
Landmarks-User-160k   | 1,262       | 2,028     | 164,172    | [Download][dl-landmarks-user] | [Visit TFF doc][tff-gldv2] |
iNaturalist-User-120k | 9,275       | 1,203     | 120,300    | [Download][dl-inat-user]      | [Visit TFF doc][tff-inat]  |
iNaturalist-Geo       | 11 to 3,606 | 1,203     | 120,300    | [Download][dl-inat-geo]       | [Visit TFF doc][tff-inat]  |
CIFAR-10              | 100         | 100       | 50,000     | [Download][dl-cifar10]        | -                          |
CIFAR-100             | 100         | 100       | 50,000     | [Download][dl-cifar100]       | -                          |

[dl-landmarks-user]: http://storage.googleapis.com/gresearch/federated-vision-datasets/landmarks-user-160k.zip
[dl-inat-user]: http://storage.googleapis.com/gresearch/federated-vision-datasets/inaturalist-user-120k.zip
[dl-inat-geo]: http://storage.googleapis.com/gresearch/federated-vision-datasets/inaturalist-geo.zip
[dl-cifar10]: http://storage.googleapis.com/gresearch/federated-vision-datasets/cifar10_v1.1.zip
[dl-cifar100]: http://storage.googleapis.com/gresearch/federated-vision-datasets/cifar100_v1.1.zip
[tff-gldv2]: https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2
[tff-inat]: https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist

Note: You may need to open link in a new browser tab when downloading.
Alternatively, you can also copy the link addresses and use a CLI tool like `wget`.

### File Format

The train and test splits are provided in different files:

*   `train` splits: `federated_train*.csv`.
*   `test` splits: `test.csv`.

The csv files contain the following columns:

*   `user_id`: Images belong to the same "user" for federated learning are
    assigned with the same id. This can be numerical ids, geo cell identifiers,
    or author names of the images. The `test` splits do not have this column.

*   `image_id`: The image ids in the source datasets. Images should be
    retrieved from the [source datasets](#source-datasets).

*   `class`: The class label used in the our federated visual classification
    paper.

*   `label`: The original class label in the source dataset (iNaturalist only).


### Tool

We provide a simple tool for parsing the csv files and outputting general statistics about the datasets.

```bash
# Example usage for inspecting the CIFAR-10 alpha-0 split.
$ python inspect_splits.py --dataset=cifar \
  --train_file=cifar10/federated_train_alpha_0.00.csv \
  --test_file=test.csv

# For detailed instructions.
$ python inspect_splits.py --help
```

## Source Datasets

We do not distribute images in this repo. If you use our split files, 
all images should be downloaded from the source datasets linked below.
These datasets and images may have different licenses and terms of use.
We do not own their copyright.

*  [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
*  [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017)
*  [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)

If you use TensorFlow Federated, datasets available as TFF datasets will
download images from the sources automatically.

## Paper

Please cite the following publication if you intend to use these datasets.

```
@inproceedings{hsu2020federated,
  author = {Tzu-Ming Harry Hsu and Hang Qi and Matthew Brown},
  title = {{Federated Visual Classification with Real-World Data Distribution}},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  month = {August},
  year = {2020}
}
```

## Changelog

* Jan 10, 2022
  * Fixed CIFAR 10/100 image ids. New zip files have a `v1.1` suffix.
  * Added links to the Landmarks and iNaturalist datasets hosted by TFF.
