# Charformer

This repository contains the Mesh-Tensorflow implementation of Charformer:
Fast Character Transformers via Gradient-based Subword Tokenization.

This implementation works with the [T5-codebase](https://github.com/google-research/text-to-text-transfer-transformer).

# Usage

Currently this codebase contains the modules/layers that can be plugged into T5 codebase. We are working on a JAX/FLAX implementation that will be later available in this repository. For now, the Mesh-TF implementation exists as a reference implementation. 


One would need to modify `transformer.py` in https://github.com/tensorflow/mesh to use
the provided Charformer layers. The code to inject Charformer layers can be found at
`https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer.py`.

### Integration Steps

Step 1: Add the following lines to the `__init__` function of Unitransformer class.

```
if self.gradient_subwords:
  tf.logging.info("Using gradient subwords..")
  self.grad_layer = [gradient_subword_layer()] * self.num_gsw_layers
```
along with new args `gradient_subwords`, `gradient_subword_layer` to the class.

Step 2: Right after the positional embeddings, add

```
if self.gradient_subwords and self.grad_layer:
  tf.logging.info("Using Charformer before computing layer stack.")
  # tensor should be batch x char_length x dim]
  for grad_layer in self.grad_layer:
    x, context = grad_layer.call(context, x)
```
Step 3:
Create a gin config (similar to the one provided in `configs/cf_v2_d3_dv_base.gin` which you may use in place of any other gin configs in the T5 codebase.

### Reference

If you use our work, or find it helpful in some form, please consider citing our paper:

```
@misc{tay2021charformer,
      title={Charformer: Fast Character Transformers via Gradient-based Subword Tokenization}, 
      author={Yi Tay and Vinh Q. Tran and Sebastian Ruder and Jai Gupta and Hyung Won Chung and Dara Bahri and Zhen Qin and Simon Baumgartner and Cong Yu and Donald Metzler},
      year={2021},
      eprint={2106.12672},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



