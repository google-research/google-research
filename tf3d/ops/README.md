# Steps to compile and prepare the sparse convolution operation

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

1. *Within the docker image*, enter `tf3d/ops` folder and run the following to test the building:

   ```bash
   # Update to a newer tensorflow version if the installed version is below 2.3.0
   # pip3 uninstall tensorflow
   # pip3 install tensorflow>=2.3.1
   ./configure.sh
   bazel run sparse_conv_ops_py_test  --experimental_repo_remote_exec
   ```

1. After the test succeeds, copy the shared library to `tf3d/ops/python` folder:

   ```bash
   cp -a bazel-bin/python/_sparse_conv_ops.so python/
   ```
1. Enter the parent folder containing the `tf3d` folder, the sparse conv ops can be imported as follows:

   ```python
   from tf3d.ops.python import sparse_conv_ops
   ```

1. Exit the docker image and the `.so` library should be compatible with the `pip` version of Tensorflow.


