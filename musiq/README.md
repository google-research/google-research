# MUSIQ: Multi-scale Image Quality Transformer

This directory contains checkpoints and model inference code for the ICCV 2021
paper:
["MUSIQ: Multi-scale Image Quality Transformer"](https://arxiv.org/abs/2108.05997)
by Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang.

*Disclaimer: This is not an official Google product.*

<img src="images/overview.png" alt="Model overview" style="width: 70%;">

## Using the models

The MUSIQ models are available on [TensorFlow Hub](https://tfhub.dev/s?q=musiq)
with documentation and a sample notebook for you to try.

But if you want to go deeper in the code, follow the instructions below.

## Pre-requisite

Install dependencies:

```
pip3 install -r requirements.txt
```

The model checkpoints can be downloaded from:
[gcloud directory link](https://console.cloud.google.com/storage/browser/gresearch/musiq)

The `./musiq` directory above contains the checkpoints for the default **MUSIQ**
model trained with 3-scale input (native resolution, 224, 384). The
`./musiq/full_size_single_scale` subdirectory contains the checkpoints
for the **MUSIQ-single** model trained with only the native resolution input.

-   **ava_ckpt.npz**: Trained on AVA dataset.
-   **koniq_ckpt.npz**: Trained on KonIQ dataset.
-   **paq2piq_ckpt.npz**: Trained on PaQ2PiQ dataset.
-   **spaq_ckpt.npz**: Trained on SPAQ dataset.
-   **imagenet_pretrain.npz**: Pretrained checkpoint on ImageNet.

## Run Inference

Default **MUSIQ** model with 3-scale input (native resolution, 224, 384):

```shell
python3 -m musiq.run_predict_image \
  --ckpt_path=/tmp/spaq_ckpt.npz \
  --image_path=/tmp/image.jpeg
```

For running the **MUSIQ-single** model, change `_SINGLE_SCALE` to `True`.

## Citation

If you find this code is useful for your publication, please cite the original
paper:

```
@inproceedings{ke2021musiq,
  title={MUSIQ: Multi-scale Image Quality Transformer},
  author={Ke, Junjie and Wang, Qifei and Wang, Yilin and Milanfar, Peyman and Yang, Feng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5148--5157},
  year={2021}
}
```
