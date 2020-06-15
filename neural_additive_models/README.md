## Neural Additive Models: Interpretable Machine Learning with Neural Nets

This repository contains open-source code
for the paper
[Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912).

<img src="https://i.imgur.com/Hvb7sb2.jpg" width="50%" alt="Neural Additive Model" >

Currently,
we release the `tf.keras.Model` for NAM which can be simply plugged into any neural network training procedure. We also provide helpers for
building a computation graph using NAM for classification/regression problems with `tf.compat.v1`.

Use `./run.sh` test script to ensure that the setup is correct.

## Dependencies

The code was tested under Ubuntu 16 and uses these packages:

- tensorflow>=1.15
- numpy>=1.15.2
- sklearn>=0.23
- pandas>=0.24
- absl-py

Citing
------
If you use this code in your research, please cite the following paper:

> Agarwal, R., Frosst, N., Zhang, X., Caruana, R., & Hinton, G. E. (2020).
> Neural additive models: Interpretable machine learning with neural nets.
> arXiv preprint arXiv:2004.13912


      @article{agarwal2020neural,
        title={Neural additive models: Interpretable machine learning with neural nets},
        author={Agarwal, Rishabh and Frosst, Nicholas and Zhang, Xuezhou and
        Caruana, Rich and Hinton, Geoffrey E},
        journal={arXiv preprint arXiv:2004.13912},
        year={2020}
      }

---

*Disclaimer: This is not an official Google product.*
