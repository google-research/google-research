# ScaNN

ScaNN (Scalable Nearest Neighbors) is a method for efficient vector similarity
search at scale. This code implements [1, 2], which includes search space
pruning and quantization for Maximum Inner Product Search and also supports
other distance functions such as Euclidean distance. The implementation is
optimized for x86 processors with AVX support. ScaNN achieves state-of-the-art
performance on [ann-benchmarks.com](http://ann-benchmarks.com) as shown on the
glove-100-angular dataset below:

![glove-100-angular](https://github.com/google-research/google-research/raw/master/scann/docs/glove_bench.png)

ScaNN can be configured to fit datasets with different sizes and distributions.
It has both TensorFlow and Python APIs. The library shows strong performance
with large datasets [1, 2]. The code is released for research purposes. For more
details on the academic description of algorithms, please see below.

References:

1. ```
   @inproceedings{avq_2020,
     title={Accelerating Large-Scale Inference with Anisotropic Vector Quantization},
     author={Guo, Ruiqi and Sun, Philip and Lindgren, Erik and Geng, Quan and Simcha, David and Chern, Felix and Kumar, Sanjiv},
     booktitle={International Conference on Machine Learning},
     year={2020},
     URL={https://arxiv.org/abs/1908.10396}
   }
   ```

1. ```
   @inproceedings{soar_2023,
     title={SOAR: Improved Indexing for Approximate Nearest Neighbor Search},
     author={Sun, Philip and Simcha, David and Dopson, Dave and Guo, Ruiqi and Kumar, Sanjiv},
     booktitle={Neural Information Processing Systems},
     year={2023},
     URL={https://arxiv.org/abs/2404.00774}
   }
   ```

## Installation

`manylinux_2_27`-compatible wheels are available on
[PyPI](https://pypi.org/project/scann/):

```
pip install scann
```

ScaNN supports Linux environments running Python versions 3.9-3.12. See
[docs/releases.md](docs/releases.md) for release notes; the page also contains
download links for ScaNN wheels prior to version 1.1.0, which were not released
on PyPI. **The x86 wheels require AVX and FMA instruction set support, while
the ARM wheels require NEON**.

In accordance with the
[`manylinux_2_27` specification](https://peps.python.org/pep-0600/), ScaNN
requires `libstdc++` version 3.4.23 or above from the operating system. See
[here](https://stackoverflow.com/questions/10354636) for an example of how
to find your system's `libstdc++` version; it can generally be upgraded by
installing a newer version of `g++`.

### Using ScaNN with TensorFlow

ScaNN has optional TensorFlow op bindings that allow ScaNN's nearest neighbor
search functionality to be embedded into a TensorFlow SavedModel. **As of
ScaNN 1.4.0, this functionality is no longer enabled by default, and `pip
install scann[tf]` is now needed to enable the TensorFlow integration.**

If `scann[tf]` is installed, the TensorFlow ops can be accessed via
`scann.scann_ops`, and the API is almost identical to `scann.scann_ops_pybind`.
For users not already in the TensorFlow ecosystem, the native Python bindings
in `scann.scann_ops_pybind` are a better fit, which is why the TensorFlow ops
are an optional extra.

#### Integration with TensorFlow Serving

We provide custom Docker images of
[TF Serving](https://github.com/tensorflow/serving) that are linked to the ScaNN
TF ops. See the [`tf_serving` directory](tf_serving/README.md) for further
information.

## Usage

See the example in [docs/example.ipynb](docs/example.ipynb). For a more in-depth
explanation of ScaNN techniques, see [docs/algorithms.md](docs/algorithms.md).

## Building from source

To build ScaNN from source, first install the build tool
[bazel](https://bazel.build) (use version 7.x), Clang 18, and libstdc++ headers
for C++17 (which are provided with GCC 9). Additionally, ScaNN requires a modern
version of Python (3.9.x or later) and Tensorflow 2.19 installed on that version
of Python. Once these prerequisites are satisfied, run the following command in
the root directory of the repository:

```
python configure.py
CC=clang-18 bazel build -c opt --features=thin_lto --copt=-mavx --copt=-mfma --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w :build_pip_pkg
./bazel-bin/build_pip_pkg
```

To build an ARM binary from an ARM machine, the prerequisites are the same, but
the compile flags are slightly modified:

```
python configure.py
CC=clang-18 bazel build -c opt --features=thin_lto --copt=-march=armv8-a+simd --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w :build_pip_pkg
./bazel-bin/build_pip_pkg
```

A .whl file should appear in the root of the repository upon successful
completion of these commands. This .whl can be installed via pip.
