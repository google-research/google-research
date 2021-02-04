# ColTran

Source code accompanying the paper [Colorization Transformer](https://openreview.net/forum?id=5NA1PinlGFu) to be presented at
ICLR 2021. Work by Manoj Kumar, Dirk Weissenborn and Nal Kalchbrenner.


<img src="coltran.png" alt="Model figure" width="600"/>

<img src="coltran_images.png" alt="Model images" width="600"/>

ColTran consists of three components, a `colorizer`, `color upsampler` and the `spatial upsampler`.

The results in the paper were reported after training each of these individual components on `4x4 TPUv2` chips. Configurations used to train models in the paper are available in the directory `configs`. Configs for extremely small models are provided at `test_configs` for testing that the model build quickly. Set the flag `--steps_per_summaries=100` to output logs quickly. Please adjust the model / batch-size while training using fewer resources.

## Requirements

```
pip install -r requirements.txt
```

## Training

Run the following command to train the colorizer

```
python -m coltran.run --config=coltran/configs/colorizer.py --mode=train --logdir=/colorizer_ckpt_dir
```
To train the color and spatial upsampler, replace `configs/colorizer.py` with
`configs/color_upsampler.py` and `configs/spatial_upsampler.py` respectively


## Evaluation

For evaluation,

```
python -m coltran.run --config=coltran/configs/colorizer.py --mode=eval_valid --logdir=/colorizer_ckpt_dir
```

## Sampling

### Single GPU Sampling

Sampling high resolution images is a three step procedure. On a P100 GPU, the colorizer samples a batch of 20 images in 3-5 minutes, while the color and spatial upsampler sample in the order of milliseconds.

Sampling configurations for each model are described by `config.sample` ConfigDict at `configs/.py`

* **sample.num_outputs** - Number of grayscale images
* **sample.logdir** - Sample write directory.
* **sample.gen_data_dir** - Path to where the previous samples are stored.
* **sample.skip_batches** - The first `skip_batches*batch_size` images from the public imagenet TF-Datasets are skipped.

Please ensure that ``num_outputs`` and ``skip_batches`` are the same across all three models.
The generated samples are written as TFRecords to `$logdir/${config.sample.logdir}`

#### Colorizer

The command samples low resolution coarsely colored 64x64 images.

```
python -m coltran.sample --config=coltran/configs/colorizer.py --mode=sample_test --logdir=/colorizer_ckpt_dir
```

#### Color Upsampler

The command converts the coarse 64x64 images from the previous step to finer 64x64 images.

**CAUTION: Please set `config.sample.gen_data_dir` of the color upsampler config to `/colorizer_ckpt_dir/${config.sample.logdir}`**

```
python -m coltran.sample --config=coltran/configs/color_upsampler.py --mode=sample_test --logdir=/cup_ckpt_dir
```

#### Spatial Upsampler

The command superresolves the previous output to high resolution 256x256 output.

**CAUTION: Please set `config.sample.gen_data_dir` of the spatial upsampler config to `$/cup_ckpt_dir/${config.sample.logdir}`**

```
python -m coltran.sample --config=coltran/configs/spatial_upsampler.py --mode=sample_test --logdir=/cup_ckpt_dir
```

### Multi GPU Sampling

When multiple GPU's are used, parallelizing sampling a the batch-level can be done easily by setting `config.sample.skip_batches` appropriately. For example, when using 2 machines and 100 images in each machine with a batch-size of 20, set `config.sample.skip_batches` of the first and second machine to 0 and 5 respectively.

## Parsing TFRecords

The generated TF Records can be converted to images easily with the following code

```
def parse_example(example_proto, res=64):
  features = {'image': tf.io.FixedLenFeature([res*res*3], tf.int64)}
  example = tf.io.parse_example(example_proto, features=features)
  image = tf.reshape(example['image'], (res, res, 3))
  return image

gen_dataset = tf.data.TFRecordDataset(listdir(path))
gen_dataset = gen_dataset.map(lambda x: parse_example(x, res))
gen_dataset = iter(gen_dataset)
for image in gen_dataset:
  plt.imshow(image)
```


## Citation

If you use the code or model please cite our paper.

```
@inproceedings{
kumar2021colorization,
title={Colorization Transformer},
author={Manoj Kumar and Dirk Weissenborn and Nal Kalchbrenner},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=5NA1PinlGFu}
}

```
