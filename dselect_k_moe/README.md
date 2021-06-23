# DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning

Code for DSelect-k as described in [DSelect-k: Differentiable Selection in the
Mixture of Experts with Applications to Multi-Task Learning](https://arxiv.org/abs/2106.03760).
Hussein Hazimeh, Zhe Zhao, Aakanksha Chowdhery, Maheswaran Sathiamoorthy, Yihua Chen, Rahul Mazumder, Lichan Hong, Ed H. Chi

If you use this codebase for your research, please cite the paper:

```
@article{hazimeh2021dselectk,
      title={DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning},
      author={Hussein Hazimeh and Zhe Zhao and Aakanksha Chowdhery and Maheswaran Sathiamoorthy and Yihua Chen and Rahul Mazumder and Lichan Hong and Ed H. Chi},
      year={2021},
      eprint={2106.03760},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Example usage of the tf.keras.layers.Layer implementation.

Construct a DSelectKGate to select 2 out of 4 experts.
```
    gate = DSelectKGate(num_nonzeros=2)
```
Output_tensor is a sparse mixture of the 4 tensors in the inputs.
```
    output_tensor = gate(inputs)
```
