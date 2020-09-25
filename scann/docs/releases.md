# ScaNN Releases

Version | Date     | Python 3.5                                                                                          | Python 3.6                                                                                          | Python 3.7
------- | -------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------
1.0.0   | 5/2/2020 | [Link](https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp35-cp35m-linux_x86_64.whl) | [Link](https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp36-cp36m-linux_x86_64.whl) | [Link](https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp37-cp37m-linux_x86_64.whl)
1.1.0   | 9/25/2020 | [Link](https://storage.googleapis.com/scann/releases/1.1.0/scann-1.1.0-cp35-cp35m-linux_x86_64.whl) | [Link](https://storage.googleapis.com/scann/releases/1.1.0/scann-1.1.0-cp36-cp36m-linux_x86_64.whl) | [Link](https://storage.googleapis.com/scann/releases/1.1.0/scann-1.1.0-cp37-cp37m-linux_x86_64.whl)

## Release Notes

### 1.1.0

**Compiled against TensorFlow 2.3.0**; this is a breaking change and ScaNN is incompatible with earier versions of TF. ScaNN 1.0.0 was compiled against TensorFlow 2.1.0.

The TensorFlow ops are now under the namespace `Scann` rather than `Addons`; any TensorFlow SavedModel using the ScaNN ops should update its [`namespace_whitelist`](https://www.tensorflow.org/api_docs/python/tf/saved_model/SaveOptions) accordingly.

ScaNN config builder changes: rather than calling `create_tf` or `create_pybind` on a `ScannBuilder` object, use the `builder()` method from scann\_ops or scann\_ops\_pybind to get a `ScannBuilder` object, and call `build()` on the object to get a searcher.

Streamlining of serialization API: the Pybind11 searcher's `load_searcher` and TensorFlow searcher's `searcher_from_module` functions now no longer require the original dataset as an argument.

Tree-AH with `dot_product` distance is 10-20% faster.

### 1.0.0

Initial Release
