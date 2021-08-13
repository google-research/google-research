# Pretrained Convolution

This repository releases pre-trained checkpoints for the ACL 2021 paper

```
Are Pretrained Convolutions better than Pretrained Transformers? (Tay et al. 2021)
```

## Usage

This repo is built with Mesh Tensorflow and T5. To use it, you'll
need to use the [T5 library](https://pypi.org/project/t5/).

The model code and implementation for [convolutions](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py)
is already released in Mesh Tensorflow. Example gin configs can be found at `./gin`.

The pretrained checkpoints released in this repository are compatible with the T5 library and passing the the model dir to `pretrained_model_dir` should be sufficient to use the convolution models.


## Checkpoints

We currently provide 3 sizes of pretrained convs, small, base and large. We provide ckpts for lightweight convs and dynamic convs. Dilated convolutions require custom MTF code so that is skipped for now since they do not perform much better over these alternatives. The entire folder can be found at `gs://pretrainedconvs`.

### LConv Small (67M parameters)

```
gs://pretrainedconvs/lconv_small
```

### LConv Base (210M parameters)

```
gs://pretrainedconvs/lconv_base
```

### LConv Large (741M parameters)

```
gs://pretrainedconvs/lconv_large
```

### DConv Base (324M parameters)


```
gs://pretrainedconvs/dconv_base
```

### DConv Large (1.2B parameters)

```
gs://pretrainedconvs/dconv_large
```

## How to Cite

If you find our work useful, please consider citing the following:

```
@misc{tay2021pretrained,
      title={Are Pre-trained Convolutions Better than Pre-trained Transformers?},
      author={Yi Tay and Mostafa Dehghani and Jai Gupta and Dara Bahri and Vamsi Aribandi and Zhen Qin and Donald Metzler},
      year={2021},
      eprint={2105.03322},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```




