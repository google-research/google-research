# Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection

This is a JAX/Flax implementation of DITO [Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection](https://arxiv.org/abs/2310.00161).

## Installation

Install the package from the root directory.

```
pip install -e .
```

## Download the DITO checkpoint and precomputed text embeddings.
Run the following commands from the root directory.

```
cd ./dito/checkpoints
./download.sh

cd ../embeddings
./download.sh
```

## Run the demo.

Run the following command from the root directory. This will run the DITO demo.

```
python ./dito/demo.py
```

You can set demo image and visualization options by the command line flags. Please refer to demo.py for more documentation on the flags.
We note that the demo model was pretrained on DataComp-1B and finetuned on the base categories of LVIS.

## Train and evaluate DITO.

Here we describe the steps to use the [COCO](https://cocodataset.org/#home) dataset for training and evaluation as an example. To use any custom dataset, users would need to follow the similar setup.

* Follow the steps [here](https://cloud.google.com/tpu/docs/tutorials/mask-rcnn-2.x#prepare-coco) to set up the COCO dataset and move it to datasets/coco. The coco directory should contain train*.tfrecord, val*.tfrecord, and instances_val2017.json (the standard COCO evaluation [file](https://cocodataset.org/#download)).

* Run the following command from the root directory:

```
OUTPUT_DIR="/your/output/dir"

./dito/train_and_eval.sh "${OUTPUT_DIR}"
```

## Set up custom datasets.

Here we describe the specific changes needed in ./dito/configs/dito_train_and_eval.gin to set up training/evaluation with custom datasets.

* Update TRAIN_FILE_PATTERN and EVAL_FILE_PATTERN to point to your dataset.
* Update TRAIN_EMBED_PATH and EVAL_EMBED_PATH to point to your cached embedding.npy.
* Update CATG_PAD_SIZE to the number of your training categories.
* Update EVAL_STEPS to the number of your validation set size.

## Citation
```
@article{kim2023dito,
  title={Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection},
  author={Dahun Kim and Anelia Angelova and Weicheng Kuo},
  journal={arXiv preprint arXiv:2310.00161},
  year={2023}
}
```

## Disclaimer
This is not an officially supported Google product.