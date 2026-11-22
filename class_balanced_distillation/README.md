# Class-Balanced Distillation for Long-Tailed Visual Recognition

This repository contains the official Tensorflow code for the following paper:
[Class-Balanced Distillation for Long-Tailed Visual Recognition](https://arxiv.org/abs/2104.05279).
Ahmet Iscen, André Araujo, Boqing Gong, Cordelia Schmid
BMVC 2021.

## Running the code

#### Setting up the Python environment

The code has been tested with Python 3.7.2. We added ImageNet-LT and iNaturalist18 datasets to the official TFDS repository as a part of this release. Therefore, we recommend using `tfds-nightly` and `tf-nightly-gpu` libraries for this code.

The following commands installs all the required libraries for this project:
```
# from google-research/
pip install -r class_balanced_distillation/requirements.txt
```

In addition to these requirements, the data-augmentation part also requires
[SimCLR](https://github.com/google-research/simclr)
to be installed and present in the `PYTHONPATH` environment variable.


#### Running the code for CBD on ImageNet-LT

The training is composed of two stages. We include pre-defined config files to faciliate running the experiments. 

To reproduce the results for `CBD`, the first stage is run with the following command:

```
# from google-research/
python -m class_balanced_distillation.run --config class_balanced_distillation/configs/stage_one_vanilla_seed1.py --workdir /home/user/class_balanced_distillation/data/models/vanilla_seed1
```

Once the training for the first stage finishes, you can run the following command for the second stage:

```
# from google-research/
python -m class_balanced_distillation.run --config class_balanced_distillation/configs/stage_two_cbd.py --workdir /home/user/class_balanced_distillation/data/models/
```

Make sure that the `model_dir` variable in `configs/stage_two_cbd.py` points to `models` folder where `vanilla_seed1` is from the first stage.


#### Running the code for CBD_ENS on ImageNet-LT

`CBD_ENS` requires multiple teacher models to be trained in the first stage. These models can be trained with the following commands:

```
# from google-research/
python -m class_balanced_distillation.run --config class_balanced_distillation/configs/stage_one_vanilla_seed1.py --workdir /home/user/class_balanced_distillation/data/models/
python -m class_balanced_distillation.run --config class_balanced_distillation/configs/stage_one_vanilla_seed2.py --workdir /home/user/class_balanced_distillation/data/models/
python -m class_balanced_distillation.run --config class_balanced_distillation/configs/stage_one_data_aug_seed1.py --workdir /home/user/class_balanced_distillation/data/models/
python -m class_balanced_distillation.run --config class_balanced_distillation/configs/stage_one_data_aug_seed2.py --workdir /home/user/class_balanced_distillation/data/models/

```

Once all the training jobs are finished, the second stage can be run with:


```
# from google-research/
python -m class_balanced_distillation.run --config class_balanced_distillation/configs/stage_two_cbd_ens.py --workdir /home/user/class_balanced_distillation/data/models/
```

Again, make sure that the `model_dir` variable in `configs/stage_two_cbd_ens.py` is the same as `--workdir` flag in the first stage.


#### Running the code on iNaturalist18

Simply change `config.dataset = "imagenet-lt"` to `config.dataset = "inaturalist18"` in all the config files.




## Reference

If you use this code, please use the following BibTeX entry to cite our work.

```
@article{iscen2021cbd,
  title={Class-Balanced Distillation for Long-Tailed Visual Recognition},
  author={Iscen, Ahmet and Araujo, Andr{\'e} and Gong, Boqing and Schmid, Cordelia},
  booktitle={The British Machine Vision Conference (BMVC)},
  year={2021}
}
```


## Questions

Please contact `iscen {at} google.com` if you find any issues with the code.
