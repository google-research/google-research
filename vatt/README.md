# VATT in Tensorflow 2

This is the official code release for [VATT: Transformers for Multimodal
Self-Supervised Learning from Raw Video, Audio and
Text](https://arxiv.org/abs/2104.11178), published in NeurIPS 2021.

## Installation

### Framework

There is a minimum framework requirement for this codebase: - `Python 3.8`,
`CUDA 10.1`, `NVIDIA Driver v 440.100`, `CuDNN 7.6.5`

### Libraries

Make sure to install the following libraries by running `pip install -r
requirements.txt`:

-   `tensorflow==2.7.0`
-   `tensorflow_addons==0.15.0`
-   `tensorflow_probability==0.15.0`
-   `tensorflow_text==2.7.0`
-   `keras==2.7.0`
-   `scikit-image`
-   `scikit-learn`
-   `scipy`
-   `six`
-   `numpy`
-   `yaml`
-   `dmvr`
-   `absl`

## Data

The data pipeline in this code is based on
[`DMVR`](https://github.com/deepmind/dmvr), which supports TF Example and TF
SequenceExample. The data loaders assume that the datasets are stored as TF Records
similar to [this](https://github.com/deepmind/dmvr/tree/master/examples)
example.

Make sure to fill in the correct constructor under `vatt/data/datasets` before
launching the main script. There is a toy example under
 `vatt/data/datasets/toy_dataset.py` for your reference.

#### Embeddings and Vocabulary

Depending on the configuration, you might need the pre-trained text embeddings and vocuabulary.
Please download [this file](https://storage.cloud.google.com/tf_model_garden/vision/vatt/misc_data.tgz) and extract it under `vatt/`.

## PreTrain

Assuming all datasets are stored and dataloaders are functioning, pre-training
can be lauched using the following: `python -m vatt.main --task=pretrain
--mode=train --model_dir=PATH/TO/RUN --model_arch=tx_fac
--strategy_type=mirrored`

If `--mode=train`, the self-supervised training will launch and if `--mode=eval`
the thorough evaluation will be launched.

The evaluation pipeline constantly loops over the `model_dir` path and looks for
new checkpoints. This means that you can launch the evaluation pipeline
separately and benefit from a continuous evaluation during the course of
pre-training.

Alternatively, you can set `--override_checkpoint=PATH/TO/CHECKPOINT` to
evaluate based on a specific checkpoint.

If you are using TPUs, you can set `--strategy_type=tpu --tpu=ADDRESS/OF/TPU`.

*   The options for `model_arch` are the following:
    *   `tx_fac`: Modality-specific VATT
    *   `ut_fac`: Modality-agnostic VATT
    *   `mmv_fac`: The CNN-based counterpart as in
        [MMV](https://arxiv.org/abs/2006.16228)

## FineTune

Once you pre-train a model, you can fine-tune the vision or audio Transformers
on a classification dataset.

Assuming all datasets are stored and dataloaders are functioning, fine-tuning
can be lauched using the following: `python -m vatt.main --task=finetune
--mode=train --model_dir=PATH/TO/RUN --model_arch=ViT_Medium
--strategy_type=mirrored`

Similarly, `mode` can take either of `train` or `eval` and a continuous
evaluation is possible by running the evaluation pipeline in parallel.

*   The options for `model_arch` are the following:
    *   `vit_base`: Vision Transformer with the `Base` configuration
    *   `vit_medium`: Vision Transformer with the `Medium` configuration
    *   `vit_large`: Vision Transformer with the `Large` configuration
    *   `wat_base`: Waveform Transformer with the `Base` configuration
    *   `wat_medium`: Waveform Transformer with the `Medium` configuration
    *   `spt_base`: Spectrogram Transformer with the `Base` configuration
    *   `spt_medium`: Spectrogram Transformer with the `Medium` configuration
    *   `i3d`: Video model based on I3D architecture
    *   `resnet2d_50`: Audio model based on a ResNet-2D architecture
        (Spectrogram-only)

In any of the settings, make sure to set the correct configuration for data and
optimization under `vatt/configs`.

## Checkpoints

### Pre-trained checkpoints

Backbone          | Model Size (Video-Audio-Text) | Checkpoint
:---------------: | :---------------------------: | :--------:
Modality Specific | Base-Base-Small               | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/pretrain/tx_fac_bbs/ckpt-500000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/pretrain/tx_fac_bbs/ckpt-500000.index)
Modality Specific | Medium-Base-Small             | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/pretrain/tx_fac_mbs/ckpt-500000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/pretrain/tx_fac_mbs/ckpt-500000.index)
Modality Specific | Large-Base-Small              | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/pretrain/tx_fac_lbs/ckpt-500000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/pretrain/tx_fac_lbs/ckpt-500000.index)
Modality Agnositc | Medium (`single-backbone`)    | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/pretrain/ut_fac_medium/ckpt-500000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/pretrain/ut_fac_medium/ckpt-500000.index)

### Fine-tuned checkpoints for Video Action Recognition

Dataset         | Model Type | Pre-trained Checkpoint     | Top-1    | Top-5 | Checkpoint
:-------------: | :--------: | :------------------------: | :------: | :---: | :--------:
Kinetics-400    | ViT Base   | Base-Base-Small            | 79.6     | 94.9  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_base/k400_modality_specific_pretrain/ckpt-100000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_base/k400_modality_specific_pretrain/ckpt-100000.index)
Kinetics-400    | ViT Medium | Medium-Base-Small          | 81.1     | 95.6  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/k400_modality_specific_pretrain/ckpt-100000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/k400_modality_specific_pretrain/ckpt-100000.index)
Kinetics-400    | ViT Large  | Large-Base-Small           | **82.1** | 95.5  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_large/k400_modality_specific_pretrain/ckpt-100000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_large/k400_modality_specific_pretrain/ckpt-100000.index)
Kinetics-400    | ViT Medium | Medium (`single-backbone`) | 79.9     | 94.9  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/k400_modality_agnostic_pretrain/ckpt-100000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/k400_modality_agnostic_pretrain/ckpt-100000.index)
Kinetics-600    | ViT Base   | Base-Base-Small            | 80.5     | 95.5  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_base/k600_modality_specific_pretrain/ckpt-100000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_base/k600_modality_specific_pretrain/ckpt-100000.index)
Kinetics-600    | ViT Medium | Medium-Base-Small          | 82.4     | 96.1  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/k600_modality_specific_pretrain/ckpt-90000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/k600_modality_specific_pretrain/ckpt-90000.index)
Kinetics-600    | ViT Large  | Large-Base-Small           | **83.6** | 96.6  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_large/k600_modality_specific_pretrain/ckpt-100000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_large/k600_modality_specific_pretrain/ckpt-100000.index)
Kinetics-600    | ViT Medium | Medium (`single-backbone`) | 80.8     | 95.5  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/k600_modality_agnostic_pretrain/ckpt-100000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/k600_modality_agnostic_pretrain/ckpt-100000.index)
Kinetics-700    | ViT Base   | Base-Base-Small            | -        | -     | TBD
Kinetics-700    | ViT Medium | Medium-Base-Small          | -        | -     | TBD
Kinetics-700    | ViT Large  | Large-Base-Small           | **72.7** | 90.5  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_large/k700_modality_specific_pretrain/ckpt-100000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_large/k700_modality_specific_pretrain/ckpt-100000.index)
Kinetics-700    | ViT Medium | Medium (`single-backbone`) | -        | -     | TBD
Moments-in-Time | ViT Base   | Base-Base-Small            | 38.7     | 67.5  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_base/mit_modality_specific_pretrain/ckpt-150000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_base/mit_modality_specific_pretrain/ckpt-150000.index)
Moments-in-Time | ViT Medium | Medium-Base-Small          | 39.5     | 68.2  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/mit_modality_specific_pretrain/ckpt-150000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/mit_modality_specific_pretrain/ckpt-150000.index)
Moments-in-Time | ViT Large  | Large-Base-Small           | **41.1** | 67.7  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_large/mit_modality_specific_pretrain/ckpt-150000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_large/mit_modality_specific_pretrain/ckpt-150000.index)
Moments-in-Time | ViT Medium | Medium (`single-backbone`) | 37.8     | 65.9  | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/mit_modality_agnostic_pretrain/ckpt-150000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/vit_medium/mit_modality_agnostic_pretrain/ckpt-150000.index)

### Fine-tuned checkpoints for Audio Event Classification

Dataset  | Model Type | Pre-trained Checkpoint     | mAP      | AUC      | d-prime   | Checkpoint
:------: | :--------: | :------------------------: | :------: | :------: | :-------: | :--------:
AudioSet | WaT Base   | Base-Base-Small            | **39.4** | **97.1** | **2.895** | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/wat_base/audioset_modality_specific_pretrain/ckpt-50000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/wat_base/audioset_modality_specific_pretrain/ckpt-50000.index)
AudioSet | WaT Medium | Medium (`single-backbone`) | 39.3     | 97.0     | 2.884     | [data](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/wat_medium/audioset_modality_agnostic_pretrain/ckpt-50000.data-00000-of-00001), [index](https://storage.cloud.google.com/tf_model_garden/vision/vatt/finetune/wat_medium/audioset_modality_agnostic_pretrain/ckpt-50000.index)

## References

```
@article{akbari2021vatt,
  title={Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text},
  author={Akbari, Hassan and Yuan, Liangzhe and Qian, Rui and Chuang, Wei-Hong and Chang, Shih-Fu and Cui, Yin and Gong, Boqing},
  journal={arXiv preprint arXiv:2104.11178},
  year={2021}
}
```


## Correspondence and Maintenance

Any feedback is appreciated. If you observed any issues, please contact us.

Corresponding author: https://github.com/hassanhub
