# Region-Aware Pretraining for Open-Vocabulary Object Detection with Vision Transformers

This is a JAX/Flax implementation of RO-ViT, the CVPR-2023 paper [Region-Aware Pretraining for Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2305.07011).
It provides the implementation of Cropped Positional Embedding (CPE) and focal contrastive loss.

## Installation

Install the package from the root directory.

```
pip install -e .
```

## Download the RO-ViT checkpoint and precomputed text embeddings.
Run the following commands from the root directory.

```
cd ./rovit/checkpoints
./download.sh

cd ../embeddings
./download.sh
```

## Run the demo.

Run the following command from the root directory. This will run the RO-ViT demo.

```
python ./rovit/demo.py
```

You can set demo image and visualization options by the command line flags. Please refer to demo.py for more documentation on the flags.
We note that the demo model was pre-trained on LAION-2B and finetuned on the base categories of LVIS.

## Citation
```
@inproceedings{kim2023region,
  title={Region-aware pretraining for open-vocabulary object detection with vision transformers},
  author={Kim, Dahun and Angelova, Anelia and Kuo, Weicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11144--11154},
  year={2023}
}
```

## Disclaimer
This is not an officially supported Google product.