# Data Preparation

We use [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for
semantic segmentation testing, evaluation and visualization on Pascal VOC,
Pascal Context and COCO datasets. You do not need to construct a tree-like data
structure within the codebase. To specify the root of each folder, you need to
modify the `data_root` attribute in each config file. All datasets include the
"background" class for mIoU evaluation.

## Semantic Segmentation

### Pascal VOC

Please follow the
[MMSegmentation Pascal VOC Preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc)
instructions to download and setup the Pascal VOC dataset.

### Pascal Context

Please refer to the
[MMSegmentation Pascal Context Preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
instructions to download and setup the Pascal Context dataset.

### COCO Object

We follow [GroupViT](https://github.com/NVlabs/GroupViT) to prepare the data of
[COCO Object](https://github.com/NVlabs/GroupViT?tab=readme-ov-file#coco).

### ADE-150 and ADE-847

We follow [FC-CLIP](https://github.com/bytedance/fc-clip) to prepare the data of
[ADE-150](https://github.com/bytedance/fc-clip/blob/main/datasets/README.md#expected-dataset-structure-for-ade20k-a150)
and
[ADE-847](https://github.com/bytedance/fc-clip/blob/main/datasets/README.md#expected-dataset-structure-for-ade20k-full-a-847).

## Referring Segmentation

### Ref-COCO, RefCOCO+ and RefCOCOg

1. Follow instructions in the [refer](https://github.com/lichengunc/refer)
directory to set up subdirectories and download annotations.

2. Download images from [COCO](https://cocodataset.org/#download).
Please use the first downloading link *2014 Train images [83K/13GB]*, and
extract the downloaded `train_2014.zip` file to
`./refer/data/images/mscoco/images`.
