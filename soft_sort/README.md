# Differentiable Ranks and Sorting operators for Tensorflow and Jax

## Overview

We propose in this work two operators that can be used to recover differentiable approximations to rank and sort vector operators. We provide JAX and TensorFlow implementations of these operators.

(Optional) Import code from Google Research repo
-------------------

```python
import glob
if 'google-research' not in ''.join(glob.glob('*')):
  !git clone https://github.com/google-research/google-research.git
import os
if os.path.basename(os.getcwd()) != 'google-research':
  os.chdir('google-research')
```

TensorFlow Examples
-------------------
```python
>>> import tensorflow as tf  
>>> import soft_sort.ops as tfops 
>>> values = tf.convert_to_tensor([[5., 1., 2.], [2., 1., 5.]], dtype=tf.float64)
>>> tfops.softsort(values, epsilon=0.1)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy=
array([[1.28653417, 1.87181597, 4.84164986],
       [1.28653417, 1.87181597, 4.84164986]])>
>>> tfops.softsort(values, epsilon=0.01)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy=
array([[1.00009997, 1.99990003, 5.        ],
       [1.00009997, 1.99990003, 5.        ]])>
>>> tfops.softranks(values, epsilon=0.1)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy=
array([[1.95319234, 0.27984306, 0.75465547],
       [0.75465547, 0.27984306, 1.95319234]])>
>>> tfops.softranks(values, epsilon=0.01)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy=
array([[2.00000000e+00, 9.60813270e-05, 9.99901972e-01],
       [9.99901972e-01, 9.60813270e-05, 2.00000000e+00]])>
>>> tfops.softquantiles(values, [.5])
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2.0000112, 1.9999999], dtype=float32)>
```    
Jax Examples
-------------------

```python
>>> import jax.numpy as np
>>> import soft_sort.jax.ops as jaxops
>>> values = np.array([[5., 1., 2.], [2., 1., 5.]], dtype=np.float32)
>>> jaxops.softsort(values, epsilon=0.1)
DeviceArray([[1.2865341, 1.8718157, 4.84165  ],
             [1.286534 , 1.8718157, 4.84165  ]], dtype=float32)
>>> jaxops.softsort(values, epsilon=0.01)
DeviceArray([[1.0000998, 1.9998999, 4.9999995],
             [1.0000998, 1.9998999, 5.       ]], dtype=float32)
>>> jaxops.softranks(values, epsilon=0.1)
DeviceArray([[1.9531922 , 0.2798431 , 0.75465536],
             [0.7546551 , 0.2798431 , 1.9531922 ]], dtype=float32)
>>> jaxops.softranks(values, epsilon=0.01)
DeviceArray([[1.9999998e+00, 9.5963478e-05, 9.9990189e-01],
             [9.9990189e-01, 9.5963478e-05, 1.9999998e+00]],dtype=float32)
>>> jaxops.softquantile(values, .5)
DeviceArray([2.0000086, 2.0000057], dtype=float32)
```



## References

Cuturi M., Teboul O., Vert JP: [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885.pdf)


[Presentation
Slides](https://drive.google.com/file/d/1J2bCRN-aN2JgTyO0zk-uviZpadRCB3Pl/view?usp=sharing)

## License

Licensed under the
[Apache 2.0](https://github.com/google-research/google-research/blob/master/LICENSE)
License.

## Disclaimer

This is not an official Google product.

