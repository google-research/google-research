# tf_cnn_benchmarks: High performance benchmarks

tf_cnn_benchmarks contains implementations of several popular convolutional
models, and is designed to be as fast as possible. tf_cnn_benchmarks supports
both running on a single machine or running in distributed mode across multiple
hosts. See the [High-Performance models
guide](https://www.tensorflow.org/performance/performance_models) for more
information.

These models utilize many of the strategies in the [TensorFlow Performance
Guide](https://www.tensorflow.org/performance/performance_guide). Benchmark
results can be found [here](https://www.tensorflow.org/performance/benchmarks).

These models are designed for performance. For models that have clean and
easy-to-read implementations, see the [TensorFlow Official
Models](https://github.com/tensorflow/models/tree/master/official).

## Getting Started

To run ResNet50 with synthetic data without distortions with a single GPU, run

```
python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=parameter_server
```

Note that the master branch of tf_cnn_benchmarks requires the latest nightly
version of TensorFlow. You can install the nightly version by running `pip
install tf-nightly-gpu` in a clean environment, or by installing TensorFlow from
source. We sometimes will create a branch of tf_cnn_benchmarks, in the form of
cnn_tf_vX.Y_compatible, that is compatible with TensorFlow version X.Y For
example, branch
[cnn_tf_v1.9_compatible](https://github.com/tensorflow/benchmarks/tree/cnn_tf_v1.9_compatible/scripts/tf_cnn_benchmarks)
works with TensorFlow 1.9.

Some important flags are

*   model: Model to use, e.g. resnet50, inception3, vgg16, and alexnet.
*   num_gpus: Number of GPUs to use.
*   data_dir: Path to data to process. If not set, synthetic data is used. To
    use Imagenet data use these
    [instructions](https://github.com/tensorflow/models/tree/master/research/inception#getting-started)
    as a starting point.
*   batch_size: Batch size for each GPU.
*   variable_update: The method for managing variables: parameter_server
    ,replicated, distributed_replicated, independent
*   local_parameter_device: Device to use as parameter server: cpu or gpu.

To see the full list of flags, run `python tf_cnn_benchmarks.py --help`.

To run ResNet50 with real data with 8 GPUs, run:

```
python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=256 \
--model=resnet50 --optimizer=momentum --variable_update=replicated \
--nodistortions --gradient_repacking=8 --num_gpus=8 \
--num_epochs=90 --weight_decay=1e-4 --data_dir=${DATA_DIR} --use_fp16 \
--train_dir=${CKPT_DIR}
```
This will train a ResNet-50 model on ImageNet with 2048 batch size on 8
GPUs. The model should train to around 76% accuracy.

## Running the tests

To run the tests, run

```bash
pip install portpicker
python run_tests.py && python run_tests.py --run_distributed_tests
```

Note the tests require portpicker.

The command above runs a subset of tests that is both fast and fairly
comprehensive. Alternatively, all the tests can be run, but this will take a
long time:

```bash
python run_tests.py --full_tests && python run_tests.py --full_tests --run_distributed_tests
```

We will run all tests on every PR before merging them, so it is not necessary
to pass `--full_tests` when running tests yourself.

To run an individual test, such as method `testParameterServer` of test class
`TfCnnBenchmarksTest` of module `benchmark_cnn_test`, run

```bash
python -m unittest -v benchmark_cnn_test.TfCnnBenchmarksTest.testParameterServer
```
