# F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models

This is a JAX/Flax demo of the ICLR-2023 paper ["F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models"](https://arxiv.org/abs/2209.15639).


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

## Download the checkpoints.
Run the following commands from the root fvlm directory. 

```
cd ./checkpoints
./download.sh
```

## Run the demo.
Run the following commands from the root fvlm directory. This will run F-VLM demo using ResNet50 backbone.

```
python3 demo.py --model=resnet_50
```

You can set model size, demo image, category string, and visualization by the command line flags. Please refer to demo.py for more documentation on the flags.

We note that the demo models are trained on a mixture of COCO, Objects365 and full LVIS dataset to increase the base category coverage. They are different from the ones used for LVIS/COCO benchmarks in the paper which are trained on subsets of LVIS/COCO vocabularies.

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