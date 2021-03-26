# **POEM**: Human **PO**se **EM**bedding

We target learning representations/embeddings for 3D human poses from their 2D projections, including:

* **`Pr-VIPE`**: Learning a view-invariant probabilistic pose embedding space. [[`paper`](https://arxiv.org/abs/1912.01001)][[`website`](https://sites.google.com/corp/view/pr-vipe/home)]
* **`CV-MIM`**: Learning disentangled view-invariant pose representations and view representations. [[`paper`](https://arxiv.org/abs/2012.01405)]

Please refer to our papers for details and consider citing them if you find the
code useful:

```
@inproceedings{sun2020view,
  title={View-Invariant Probabilistic Embedding for Human Pose},
  author={Sun, Jennifer J and Zhao, Jiaping and Chen, Liang-Chieh and Schroff, Florian and Adam, Hartwig and Liu, Ting},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{zhao2020learning,
  title={Learning View-Disentangled Human Pose Representation by Contrastive Cross-View Mutual Information Maximization},
  author={Zhao, Long and Wang, Yuxiao and Zhao, Jiaping and Yuan, Liangzhe and Sun, Jennifer J and Schroff, Florian and Adam, Hartwig and Peng, Xi and Metaxas, Dimitris and Liu, Ting},
  booktitle={CVPR},
  year={2020}
}
```

## Getting Started
The installation requires [Python3](https://www.python.org/), [virtualenv](https://virtualenv.pypa.io/), and [pip](https://pip.pypa.io/) pre-installed.
Please refer to [`requirements.txt`](https://github.com/google-research/google-research/blob/master/poem/requirements.txt) and
[`run.sh`](https://github.com/google-research/google-research/blob/master/poem/run.sh)
for how to install the required packages.

**Note:** The [`run.sh`](https://github.com/google-research/google-research/blob/master/poem/run.sh) script only provides a sample
command for how to launch a training job on a small dummy data table. Running it
as is does NOT launch training job on any real datasets or generate any
meaningful model checkpoints.

This repository is organized with the following folders:

* [`pr_vipe`](https://github.com/google-research/google-research/tree/master/poem/pr_vipe): Pr-VIPE project code.
* [`cv_mim`](https://github.com/google-research/google-research/tree/master/poem/cv_mim): CV-MIM project code.
* [`core`](https://github.com/google-research/google-research/tree/master/poem/core): Common utility libraries.
* [`tools`](https://github.com/google-research/google-research/tree/master/poem/tools): Common utility tools.
* [`testdata`](https://github.com/google-research/google-research/tree/master/poem/testdata): Data for code testing (not for real training/evaluation).
* [`doc`](https://github.com/google-research/google-research/tree/master/poem/doc): Documentation files.

## Getting Help
Please report issues related to this repository to the [tracker](https://github.com/google-research/google-research/issues) and make sure to
include the **"`[POEM]`"** prefix in the issue title.

## Contact
- [Ting Liu](https://github.com/tingliu)
- [Jennifer J. Sun](https://github.com/jenjsun)
- [Long Zhao](https://github.com/garyzhao)
- [Liangzhe Yuan](https://github.com/yuanliangzhe)

## Updates
- `03/25/2021`: Moved **`Pr-VIPE`** code into a [subfolder](https://github.com/google-research/google-research/tree/master/poem/pr_vipe). Added **`CV-MIM`** [subfolder](https://github.com/google-research/google-research/tree/master/poem/cv_mim) and updated documentations.
- `03/17/2021`: Fixed an [issue](https://github.com/google-research/google-research/issues/636) in camera augmentation.
- `03/04/2021`: Added a program for running model inference.
- `10/21/2020`: Added cross-view pose retrieval evaluation frame keys.
- `10/15/2020`: Added training TFRecords generation program.
- `07/02/2020`: First release. Included core TensorFlow code for model training.
