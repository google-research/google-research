# Scaling Efficiently: Insights from Pre-training and Finetuning Transformers

This github releases checkpoints for the "Scaling Efficiently" paper.
These checkpoints make use of Mesh Tensorflow and the T5 library and
one should be able to load them in the T5 library/codebase.

## Download checkpoints

The checkpoints are found in the following GCP buckets.

In total, there are about 170 trained models each in their own individual
folders. The details of each model (hyperparameters) can be found
in their `operative_config.gin` file. The checkpoints are hosted
at the following GCP bucket.

```
gs://scenic-bucket/scaling_explorer/scaling_explorer
```

You'll be able to find folders written in the notation: `bi_v1_{SIZE}_{OPS}_{DATE}`.
For example `bi_v1_3B_l12_law_03-20-22-27` refers to taking a 3B model,
shrinking the layers to 12 layers. 

In each folder, we release checkpoints at 100K, 200K, 300K, 400K and 524K
steps for richer analysis. A caveat is that due to some training failures,
some intermediate checkpoints may not be availiable for a small number
of runs.


## Usage

These checkpoints are compatible with the [T5 library](https://github.com/google-research/text-to-text-transfer-transformer).
You can pass the checkpoint into `--model_dir` of the `t5_mesh_transformer` binary to use it.
Note: we did not test it with the public T5 codebase on Cloud and rather only on our internal systems. However, it should
still work though.


## Reference

If you enjoy our work of find it useful, we would appreciate it if you could
cite our paper at:


```
@article{tay2021scale,
  title={Scale efficiently: Insights from pre-training and fine-tuning transformers},
  author={Tay, Yi and Dehghani, Mostafa and Rao, Jinfeng and Fedus, William and Abnar, Samira and Chung, Hyung Won and Narang, Sharan and Yogatama, Dani and Vaswani, Ashish and Metzler, Donald},
  journal={arXiv preprint arXiv:2109.10686},
  year={2021}
}
```
