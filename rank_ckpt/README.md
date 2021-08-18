# Ranking Neural Checkpoints

Disclaimer: This is not an official Google product.

## Overview

This directory contains code and checkpoints described in
[Ranking Neural Checkpoints (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Ranking_Neural_Checkpoints_CVPR_2021_paper.pdf).
### Abstract
*This paper is concerned with ranking many pre-trained deep neural networks (DNNs), called checkpoints, for the transfer learning to a downstream task. Thanks to the broad use of DNNs, we may easily collect hundreds of checkpoints from various sources. Which of them transfers the best to our downstream task of interest? Striving to answer this question thoroughly, we establish a neural checkpoint ranking benchmark (NeuCRaB) and study some intuitive ranking measures. These measures are generic, applying to the checkpoints of different output types without knowing how the checkpoints are pre-trained on which datasets. They also incur low computation cost, being practically meaningful. Our results suggest that the linear separability of the features extracted by the checkpoints is a strong indicator of transferability. We also arrive at a new ranking measure, $\mathcal{N}$LEEP, which gives rise to the best performance in the experiments. Code will be made publicly available.*



For questions or issues, contact yandongli@google.com.

## Neural Checkpoint Ranking Benchmark (NeuCRaB)
We establish a neural checkpoint ranking benchmark (NeuCRaB) to study the problem systematically. We list three groups of neural checkpoints as follow:
Please download the [*checkpoints*](http://storage.googleapis.com/gresearch/rank_ckpt/rank_ckpt.zip) and unzip them, All the checkpoints are under `rank_ckpt`. (Please copy the [*link*](http://storage.googleapis.com/gresearch/rank_ckpt/rank_ckpt.zip) to a new browser window so that you can download the zip file properly).

### Group I: Checkpoints of mixed supervision
- [*WAE-UKL*](https://tfhub.dev/vtab/wae-ukl/1).
- [*WAE-GAN*](https://tfhub.dev/vtab/wae-gan/1).
- [*Cond-BigGAN*](https://tfhub.dev/vtab/cond-biggan/1).
- [*WAE-MMD*](https://tfhub.dev/vtab/wae-mmd/1).
- [*VAE*](https://tfhub.dev/vtab/vae/1).
- [*Uncond-BigGAN*](https://tfhub.dev/vtab/uncond-biggan/1).
- [*Jigsaw*](https://tfhub.dev/vtab/jigsaw/1).
- [*Rel.Pat.Loc*](https://tfhub.dev/vtab/relative-patch-location/1).
- [*Exemplar*](https://tfhub.dev/vtab/exemplar/1).
- [*Rotation*](https://tfhub.dev/vtab/rotation/1).
- [*Semi-Rotation-10%*](https://tfhub.dev/vtab/semi-rotation-10/1).
- [*Semi-Exemplar-10%*](https://tfhub.dev/vtab/semi-exemplar-10/1).
- [*Sup-100%*](https://tfhub.dev/vtab/sup-100/1).
- [*Sup-Exemplar-100%*](https://tfhub.dev/vtab/sup-exemplar-100/1).
- *Sup-100%-Inat*: `rank_ckpt/inat_res50/model.ckpt-300000`.
- *Sup-100%-Pla*: `rank_ckpt/places_res50/model.ckpt-300000`.

### Group II: Checkpoints at different pre-training stages
- *Img-90k*: `rank_ckpt/imagenet_res50/model.ckpt-90000`.
- *Img-180k*: `rank_ckpt/imagenet_res50/model.ckpt-180000`.
- *Img-270k*: `rank_ckpt/imagenet_res50/model.ckpt-270000`.
- *Img-300k*: `rank_ckpt/imagenet_res50/model.ckpt-300000`.
- *Inat-90k*: `rank_ckpt/inat_res50/model.ckpt-90000`.
- *Inat-180k*: `rank_ckpt/inat_res50/model.ckpt-180000`.
- *Inat-270k*: `rank_ckpt/inat_res50/model.ckpt-270000`.
- *Inat-300k*: `rank_ckpt/inat_res50/model.ckpt-300000`.
- *Pla-60k*: `rank_ckpt/places_res50/model.ckpt-60000`.
- *Pla-120k*: `rank_ckpt/places_res50/model.ckpt-120000`.
- *Pla-180k*: `rank_ckpt/places_res50/model.ckpt-180000`.
- *Pla-200k*: `rank_ckpt/places_res50/model.ckpt-200000`.

### Group III: Checkpoints of heterogeneous architectures
- *Inception-ResNet-v2*: `rank_ckpt/imagenet_inception_resnet_v2/model.ckpt-90000`.
- *Inception-v1*: `rank_ckpt/imagenet_inception_v1/model.ckpt-300000`.
- *Inception-v2*: `rank_ckpt/imagenet_inception_v2/model.ckpt-300000`.
- *Inception-v3*: `rank_ckpt/imagenet_inception_v3/model.ckpt-300000`.
- *Inception-v4*: `rank_ckpt/imagenet_inception_v4/model.ckpt-300000`.
- *MobileNet-v1*: `rank_ckpt/imagenet_mobilenet_v1/model.ckpt-300000`.
- *MobileNet-v1-025*: `rank_ckpt/imagenet_mobilenet_v1_025/model.ckpt-300000`.
- *MobileNet-v2*: `rank_ckpt/imagenet_mobilenet_v2/model.ckpt-300000`.
- *MobileNet-v2-035*: `rank_ckpt/imagenet_mobilenet_v2_035/model.ckpt-300000`.
- *MobileNet-v3-small*: `rank_ckpt/imagenet_mobilenet_v3_small/model.ckpt-300000`.
- *MobileNet-v3-large*: `rank_ckpt/imagenet_mobilenet_v3_large/model.ckpt-300000`.
- *ResNet-v1-50*: `rank_ckpt/imagenet_resnet_v1_50/model.ckpt-300000`.
- *ResNet-v1-101*: `rank_ckpt/imagenet_resnet_v1_101/model.ckpt-300000`.

## Usage
You can directly use [*Visual Task Adaptation Benchmark (VTAB)*](https://github.com/google-research/task_adaptation) to finetune the checkpoints from tf-hub on different downstream tasks (Caltech101, Flowers102, Camelyon, Sun397). 

For the other checkpoints, you can simply replace the network by changing the [*line 47-52*](https://github.com/google-research/task_adaptation/blob/master/task_adaptation/model.py#L47) with the following snippet:
```
net_fn = nets_factory.get_network_fn(
        arch_name,
        num_classes=0,
        weight_decay=0.0001,
        is_training=is_trainable)
pre_logits, _ = net_fn(features["image"])
```
where arch_name can be found in [*nets_factory*](https://github.com/tensorflow/models/blob/master/research/slim/nets/nets_factory.py)

And then initilize the model by `scaffold` funtion:
```
all_variables_to_restore = []
for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
  all_variables_to_restore.append(var)

input_checkpoint = '/path/to/your/checkpoint'
pretrain_saver = tf.train.Saver(all_variables_to_restore)
def init_fn(scaffold, session):
    pretrain_saver.restore(session, input_checkpoint)
scaffold = tf.train.Scaffold(init_fn=init_fn)
return tf.estimator.EstimatorSpec(
    mode=mode, predictions=predictions, scaffold=scaffold)
 ```

## Note
If you use this code, please consider adding the corresponding citation:

```
@InProceedings{Li_2021_CVPR,
    author    = {Li, Yandong and Jia, Xuhui and Sang, Ruoxin and Zhu, Yukun and Green, Bradley and Wang, Liqiang and Gong, Boqing},
    title     = {Ranking Neural Checkpoints},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2663-2673}
}

```
