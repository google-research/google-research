# F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models

This is the JAX/FLAX implementation of the ICLR-2023 paper ["F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models"](https://arxiv.org/abs/2209.15639).
The model is also supported on the Cloud Vertex API [here](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/109), where you can train and predict with this model on Google Cloud Vertex AI Training and Prediction service using the provided notebook at the top of the model card.


## Installation
We use the Python built-in virtual env to set up the environment. Run the following commands:

```
svn export https://github.com/google-research/google-research/trunk/fvlm

PATH_TO_VENV=/path/to/your/venv
python3 -m venv ${PATH_TO_VENV}
source ${PATH_TO_VENV}/bin/activate
```

Install the requirements from the root fvlm directory.

```
pip install -r requirements.txt
```

For GPU training, please refer to [this github page](https://github.com/jax-ml/jax/issues/13637) for installation instructions.

## Download the checkpoints.
Run the following commands from the root fvlm directory. 

```
cd ./checkpoints
./download.sh
```

For users who want to use the FLAX checkpoints directly rather than tf.SavedModel, we have prepared the checkpoints for downloading by the commands below:

```
MODEL="r50"  # Supported model: r50, r50x4, r50x16
wget "https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fvlm/jax_checkpoints/${MODEL}_checkpoint_184000"
```
We recommend users to run the above commands in the checkpoints directory.

## Run the demo.
Run the following commands from the root fvlm directory. This will run F-VLM demo using ResNet50 backbone.

```
python3 demo.py --model=resnet_50
```

You can set model size, demo image, category string, and visualization by the command line flags. Please refer to demo.py for more documentation on the flags.

We note that the demo models are trained on a mixture of COCO, Objects365 and full LVIS dataset to increase the base category coverage. They are different from the ones used for LVIS/COCO benchmarks in the paper which are trained on subsets of LVIS/COCO vocabularies.

## Export the JAX checkpoint.
Run the following commands from the root fvlm directory.

```
INPUT_DIR="/your/input/jax_checkpoint_dir"
OUTPUT_DIR="/your/output/savedmodel/dir"

./scripts/export_model.sh "${INPUT_DIR}" "${OUTPUT_DIR}"
```

## Train and evaluate F-VLM from CLIP checkpoints.

Here we describe the steps to use the [COCO](https://cocodataset.org/#home) dataset for training and evaluation as an example. To use any custom dataset, users would need to follow the same setup.

* Follow the steps [here](https://cloud.google.com/tpu/docs/tutorials/mask-rcnn-2.x#prepare-coco) to set up the COCO dataset and move it to datasets/coco. The coco directory should contain train*.tfrecord, val*.tfrecord, and instances_val2017.json (the standard COCO evaluation [file](https://cocodataset.org/#download)).
* Download the precomputed COCO category embeddings and store it by running

```
./scripts/download_precomputed_embeddings.sh
```

* Run the following command:

```
OUTPUT_DIR="/your/output/dir"

./train_and_eval.sh "${OUTPUT_DIR}"
```

## Set up custom datasets.

Here we describe the specific changes needed in ./scripts/fvlm_train_and_eval.gin to set up training/evaluation with custom datasets.

* Update TRAIN_FILE_PATTERN and EVAL_FILE_PATTERN to point to your dataset.
* Update EMBED_PATH to point to your cached embedding.npy.
* Update CATG_PAD_SIZE to the number of your training categories.
* Update EVAL_STEPS to the number of your validation set size.

## Citation
```
@inproceedings{
kuo2023openvocabulary,
title={Open-Vocabulary Object Detection upon Frozen Vision and Language Models},
author={Weicheng Kuo and Yin Cui and Xiuye Gu and AJ Piergiovanni and Anelia Angelova},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=MIMwy4kh9lf}
}
```

## Demo Image Source and License

Source: https://pixabay.com/nl/photos/het-fruit-eten-citroen-limoen-3134631/

Creative Commons License: https://pixabay.com/nl/service/terms/

## Disclaimer
This is not an officially supported Google product.