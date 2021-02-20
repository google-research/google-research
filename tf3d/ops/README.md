# Preparing the 3D Sparse Convolution Op
  You should be able to use the pre-compiled package if you have the following settings. Otherwise, please compile the op as a shared library imported by Python, and/or as a `wheel` package.
  
## Using the pre-compiled package
   If your environment is `Python 3.6` or `3.7` and `manylinux2010_x86_64` platform, you may install the `wheel` package in `tf3d/ops/packages` folder.
   
```bash
   # Note that the wheel has a requirement of 'tensorflow >= 2.3.0'.
   # for python 3.6
   pip3 install tf3d/ops/packages/tensorflow_sparse_conv_ops-0.0.1-cp36-cp36m-linux_x86_64.whl
   # for python 3.7
   pip3 install tf3d/ops/packages/tensorflow_sparse_conv_ops-0.0.1-cp37-cp37m-linux_x86_64.whl   
```
   
## Compile the ops within a docker image

1. Download the [Tensorflow repo](https://github.com/tensorflow/tensorflow) to a local folder `tensorflow`.
1. Create a folder at `tf3d/ops/third_party`, and copy the following files / folder from `tensorflow/third_party` folder:
   * `eigen3`
   * `mkl`
   * `toolchains`
   * `BUILD`
   * `com_google_absl_fix_mac_and_nvcc_build.patch`
   * `com_google_absl.BUILD`
   * `cub.BUILD`
   * `eigen.BUILD`
   * `repo.bzl`
1. Download the [TensorFlow Custom Op repo](https://github.com/tensorflow/custom-op) to a local folder `tf_custom_op`.
1. Copy the files / folders to `tf3d/ops` folder from `tf_custom_op` folder:
   * `gpu`
   * `tf`
   * `configure.sh`

   The above steps can be done with the following commands:

    ```bash
    git clone https://github.com/tensorflow/tensorflow --depth=1
    git clone https://github.com/tensorflow/custom-op --depth=1
    export TF_FOLDER="PATH_TO_TF_REPO_FOLDER"
    export CUSTOM_OP_FOLDER="PATH_TO_CUSTOM_OP_REPO_FOLDER"

    mkdir -p tf3d/ops/third_party
    cp -a ${TF_FOLDER}/third_party/eigen3 ${TF_FOLDER}/third_party/mkl \
    ${TF_FOLDER}/third_party/toolchains ${TF_FOLDER}/third_party/BUILD \
    ${TF_FOLDER}/third_party/eigen.BUILD \
    ${TF_FOLDER}/third_party/com_google_absl_fix_mac_and_nvcc_build.patch \
    ${TF_FOLDER}/third_party/com_google_absl.BUILD \
    ${TF_FOLDER}/third_party/cub.BUILD ${TF_FOLDER}/third_party/repo.bzl \
    tf3d/ops/third_party/
    cp -a ${CUSTOM_OP_FOLDER}/gpu ${CUSTOM_OP_FOLDER}/tf \
    ${CUSTOM_OP_FOLDER}/configure.sh tf3d/ops/
    ```

1. Following the [TensorFlow Custom Op repo](https://github.com/tensorflow/custom-op)'s guidance, set up the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and enter the docker image `2.3.0-custom-op-gpu-ubuntu16`.

   Run the following to enter the docker image with mapped `tf3d` folder.
   
   ```bash
   docker pull tensorflow/tensorflow:2.3.0-custom-op-gpu-ubuntu16
   docker run --runtime=nvidia --privileged  -it -v ${PATH_TO_FOLDER_WITH_TF3D}:/working_dir -w /working_dir  tensorflow/tensorflow:2.3.0-custom-op-gpu-ubuntu16
   ```
   
1. *Within the docker image*, enter `tf3d/ops` folder and run the following to test the building:

   ```bash
   # Change the tensorflow version to 2.3.0
   pip3 uninstall tensorflow
   pip3 install tensorflow==2.3.0
   ./configure.sh
   bazel run sparse_conv_ops_py_test  --experimental_repo_remote_exec
   ```

1. After the test succeeds, copy the shared library to `tf3d/ops/tensorflow_sparse_conv_ops` folder:

   ```bash
   cp -a bazel-bin/tensorflow_sparse_conv_ops/_sparse_conv_ops.so tensorflow_sparse_conv_ops/
   ```
   
   The `.so` library should be compatible with  the `pip` version of Tensorflow.

1. *Optional*: Build a `wheel` package to be installed by `pip`.

   ```bash
   chmod +x build_pip_pkg.sh
   # modify build_pip_pkg if you want a wheel for a specific python3 version.
   bazel build :build_pip_pkg   --experimental_repo_remote_exec
   bazel-bin/build_pip_pkg artifacts
   # The wheel file is generated at
   # artifacts/tensorflow_sparse_conv_ops-0.0.1-cp36-cp36m-linux_x86_64.whl
   ```

1. Exit the docker image.

   Enter the parent folder containing the `tf3d` folder, the sparse conv ops can be imported as follows:

   ```python
   import tf3d.ops.tensorflow_sparse_conv_ops as sparse_conv_ops
   ```


   If you have compiled the `wheel` package, it can be installed via `pip3 install PATH_TO_COMPILED_WHEEL` and imported by `import tensorflow_sparse_conv_ops as sparse_conv_ops`.


## FAQ:

1. What if I see the "undefined symbol" error when importing the ops?

   Make sure to use Tensorflow 2.3.0 version when compiling the ops (during the linking stage, the symbol is found in `${PATH_TO_PYTHON_LIB}/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so`). When using the ops, make sure to use Tensorflow version 2.3.x. Tensorflow 2.4.x has not been tested yet.

  Detailed error:
  
  ```bash
  tensorflow.python.framework.errors_impl.NotFoundError: /root/.cache/bazel/_bazel_root/ec891c5b3b8ae1c73a1e1d73216b2747/execroot/__main__/bazel-out/k8-opt/bin/sparse_conv_ops_py_test.runfiles/__main__/tensorflow_sparse_conv_ops/_sparse_conv_ops.so: undefined symbol: _ZN10tensorflow8OpKernel11TraceStringEPNS_15OpKernelContextEb
  ```


