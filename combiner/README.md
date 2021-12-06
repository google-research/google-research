# Implementation of Combiner: Full Attention Transformer with Sparse Computation Cost

Paper: https://arxiv.org/abs/2107.05768

This code implements Combiner. The following demonstration shows the core implementation with a toy task training.

# Install

Navigate to the root of project, and perform:

    pip install -r requirements.txt
    pip install -e .

# Example usage

We provide both the tensorflow and jax implementations of the core components.

## JAX implementation

Navigate to `jax/` folder, and run

    python main.py -model {JAX_MODEL_NAME}

where the `JAX_MODEL_NAME` can be selected from `{axial, mixture, fixed, logn}`.

## TF implementation

Navigate to `tf/` folder, and run

    python main.py --hparam_set={TF_MODEL_NAME}

where the `TF_MODEL_NAME` can be selected from `{AxialRowMajorLM, SqrtFixedFullLM, AxialMixtureLM}`.


# Disclaimer

This is not an officially supported Google product.

Contact hyren@cs.stanford.edu and hadai@google.com for questions.


# Reference

```
@article{ren2021combiner,
  title={Combiner: Full attention transformer with sparse computation cost},
  author={Ren, Hongyu and Dai, Hanjun and Dai, Zihang and Yang, Mengjiao and Leskovec, Jure and Schuurmans, Dale and Dai, Bo},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
