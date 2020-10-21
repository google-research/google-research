# **POEM**: Human **PO**se **EM**bedding

We propose an approach for learning a compact view-invariant probabilistic
embedding space for 3D human pose from their 2D projections.

<p align="center">
  <img src="doc/fig/manifold.png" width=800></br>
</p>

Please refer to our paper for details and consider citing it if you find the
code useful:

```
@inproceedings{vipoem2020,
  title={View-Invariant Probabilistic Embedding for Human Pose},
  author={Sun, Jennifer J and Zhao, Jiaping and Chen, Liang-Chieh and Schroff, Florian and Adam, Hartwig and Liu, Ting},
  booktitle={ECCV},
  year={2020}
}
```

## Getting Started
The installation requires [Python3](https://www.python.org/), [virtualenv](https://virtualenv.pypa.io/), and [pip](https://pip.pypa.io/) pre-installed.
Please refer to [`requirements.txt`](https://github.com/google-research/google-research/blob/master/poem/requirements.txt) and
[`run.sh`](https://github.com/google-research/google-research/blob/master/poem/run.sh)
for how to install the required packages.

## Getting Help
Please report issues related to this repository to the [tracker](https://github.com/google-research/google-research/issues) and make sure to
include the **"[POEM]"** prefix in the issue title.

## Contact
- [Ting Liu](https://github.com/tingliu)
- [Jennifer J. Sun](https://github.com/jenjsun)

## Updates
- `10/21/2020`: Added cross-view pose retrieval evaluation frame keys.
- `10/15/2020`: Added training TFRecords generation program.
- `07/02/2020`: First release. Included core TensorFlow code for model training.
