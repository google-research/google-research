# ScaNN Release Notes

### 1.2.4
Updated to compile against TensorFlow 2.7.0; **not** backwards-compatible with earlier versions of TensorFlow. Python 3.6 support has been dropped because TF 2.7 doesn't support Python 3.6.

### 1.2.3
Updated to compile against TensorFlow 2.6.0; **not** backwards-compatible with earlier versions of TensorFlow.

### 1.2.2
Added support for Python 3.9. Wheels and code now depend on TensorFlow 2.5.0. **Code no longer compiles against TensorFlow 2.4.x or earlier due to a change in TensorFlow's Abseil C++ dependency.**

### 1.2.1
Improved default parameters to the `tree()` method of `scann_builder`.

### 1.2.0
Python 3.5 support has been dropped, allowing ScaNN to use f-strings and other Python 3.6+ features in its code.

Wheels are now built against TensorFlow 2.4.0. **These wheels are incompatible with TensorFlow 2.3.x**; continue using ScaNN 1.1.1 or compile ScaNN yourself if you need to use ScaNN with TensorFlow 2.3.x.

Errors now include more debugging information and context.

Passing `quantize=True` to the `tree(...)` of ScaNN builder now works with squared L2 distance; it previously only supported dot products.

### 1.1.1

`manylinux2014` wheels are now available on pip, and a Python 3.8 wheel has been added. These wheels only require `libstdc++` version 3.4.19 and therefore should have greater operating system compatibility. Previous releases were not `manylinux2014` compatible and required `libstdc++` version 3.4.26 or greater.

### 1.1.0

**Compiled against TensorFlow 2.3.0**; this is a breaking change and ScaNN is incompatible with earier versions of TF. ScaNN 1.0.0 was compiled against TensorFlow 2.1.0.

The TensorFlow ops are now under the namespace `Scann` rather than `Addons`; any TensorFlow SavedModel using the ScaNN ops should update its [`namespace_whitelist`](https://www.tensorflow.org/api_docs/python/tf/saved_model/SaveOptions) accordingly.

ScaNN config builder changes: rather than calling `create_tf` or `create_pybind` on a `ScannBuilder` object, use the `builder()` method from scann\_ops or scann\_ops\_pybind to get a `ScannBuilder` object, and call `build()` on the object to get a searcher.

Streamlining of serialization API: the Pybind11 searcher's `load_searcher` and TensorFlow searcher's `searcher_from_module` functions now no longer require the original dataset as an argument.

Tree-AH with `dot_product` distance is 10-20% faster.

### 1.0.0

Initial Release

## ScaNN wheel archive

**The latest version of ScaNN is available on pip; the following table is no longer updated and only provides outdated binaries!** Prior to version 1.1.1, ScaNN wheels were not `manylinux2014` compatible and weren't distributed through pip. These older wheels may be downloaded below.

Version | Date     | Python 3.5                                                                                          | Python 3.6                                                                                          | Python 3.7
------- | -------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------
1.0.0   | 5/2/2020 | [Link](https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp35-cp35m-linux_x86_64.whl) | [Link](https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp36-cp36m-linux_x86_64.whl) | [Link](https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp37-cp37m-linux_x86_64.whl)
1.1.0   | 9/25/2020 | [Link](https://storage.googleapis.com/scann/releases/1.1.0/scann-1.1.0-cp35-cp35m-linux_x86_64.whl) | [Link](https://storage.googleapis.com/scann/releases/1.1.0/scann-1.1.0-cp36-cp36m-linux_x86_64.whl) | [Link](https://storage.googleapis.com/scann/releases/1.1.0/scann-1.1.0-cp37-cp37m-linux_x86_64.whl)
