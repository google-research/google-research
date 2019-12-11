# Codebase for "Distance-Based Learning from Errors for Confidence Calibration."

Authors: Chen Xing, Sercan O. Arik, Zizhao Zhang, Tomas Pfister

Link: https://arxiv.org/abs/1912.01730

We propose distance-based learning from errors (DBLE) to improve confidence
calibration of DNNs. DBLE bases its confidence estimation on distances in the
representation space.  We first adapt prototypical learning for training of a
classification model for DBLE. It yields a representation space where the
distance from a test sample to its ground-truth class center can calibrate the
model performance. At inference, however, these distances are not available due
to the lack of ground-truth labels. To circumvent this by approximately
inferring the distance for every test sample, we propose to train a confidence
model jointly with the classification model by merely learning from
mis-classified training samples, which we show to be highly beneficial for
effective learning. On multiple datasets and architectures, we demonstrate that
DBLE outperforms alternative single-modal confidence calibration approaches.

## Model architectures

DBLE is model agnostic and can be applied to classification task on any
datasets. In this directory, we provide example implemetations for 3
widely-used neural network architectures:
* MLP
* VGG
* ResNet

Detailed model architectures can be specified by modifying hyper-parameters of
every model architecture. For example, in dble.py, we initialize ResNet-50 for
CIFAR dataset as,

```
resnet_model = resnet.Model( wd=flags.weight_decay,
resnet_size=50, bottleneck=True, num_classes=flags.num_classes_train,
num_filters=16, kernel_size=3, conv_stride=1, first_pool_size=None,
first_pool_stride=None, block_sizes=[8, 8, 8], block_strides=[1, 2, 2],
data_format='channels_last')
```

If other model architectures except MLP, VGG or ResNet are required, the class
implementation of the architectures should be put into './new_arch.py'.
'new_arch' can be replaced by the name of the new architecture.

In class new_arch, two functions encoder(self, inputs, training) and
confidence_model(self, mu, training) should be implemented.

encoder(self, inputs, training) takes the raw samples as input and output their
representations. The encoder architecture should have the appropriate inductive 
bias for the input data type, e.g. ResNet for images.

confidence_model(self, mu, training) takes the sample representations as input 
and output their variances for confidence calibration. The confidence_model
architecture can be a simple low capacity model, such as few MLP layers.

## Data-sets

DBLE can be applied to classification tasks for confidence calibration on any
datasets. In this directory, we demonstrate DBLE on
4 image classification data-sets,
* MNIST
* CIFAR-10
* CIFAR-100
* Tiny-ImageNet

The pre-processing of data-sets is in 'data_loader.py'. For example,
the loading and pre-processing of CIFAR-10 is,

```
def _load_cifar10():
(x_train, y_train), (x_test, y_test) = cifar10.load_data() x_train =
x_train.astype('float32') x_test = x_test.astype('float32') fields = x_train,
np.squeeze(y_train) fields_test = x_test, np.squeeze(y_test)

return fields, fields_test ` ` def augment_cifar(batch_data, is_training=False):
image = batch_data if is_training: image =
tf.image.resize_image_with_crop_or_pad(batch_data, 32 + 8, 32 + 8) i =
image.get_shape().as_list()[0] image = tf.random_crop(image, [i, 32, 32, 3])
image = tf.image.random_flip_left_right(image) image =
tf.image.per_image_standardization(image)

return image
```

If other data-sets are required to test DBLE, 2 functions _load_new_data() and
augment_new_data(batch_data, is_training=False) should be implemented.
'new_data' can be replaced by the name of the new data-set.

_load_new_data() reads the raw samples and labels of the training and the test
set and returns two tuples of arrays, fields and
fields_test. Each tuple contains the array of samples and the
corresponding labels. The first dimensions of the two arrays should
be equal.

augment_cifar(batch_data, is_training=False) takes the tensor of a batch of
samples as input and output the processed batch of samples.

## Hyper-parameters of DBLE and command examples

All hyper-parameters required for the training and evaluation of DBLE are listed
in 'main_dble.py'. Here we list some main hyper-parameters that should be
adjusted with respect to the data sets and model architectures used.

```
parser.add_argument('--model_name', type=str, default='vgg')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument( '--num_cases_train', type=int, default=50000)
parser.add_argument( '--num_cases_test', type=int, default=10000)
parser.add_argument( '--num_samples_per_class', type=int, default=5000, help='')
parser.add_argument( '--num_classes_total', type=int, default=10, help='Number
of classes in total of the data set.' ) parser.add_argument(
'--num_classes_test', type=int, default=10, help='Number of classes in the test
phase. ') parser.add_argument( '--num_classes_train', type=int, default=10,
help='Number of classes in a protoypical episode.' ) parser.add_argument(
'--num_shots_train', type=int, default=10, help='Number of shots (supports) in a
prototypical episode.') parser.add_argument( '--train_batch_size', type=int,
default=100) parser.add_argument('--weight_decay', type=float, default=0.0005)
```

Here is the command example for the experiment that classifies CIFAR-10 with
VGG-11.

```
python3 -m main_dble
--num_classes_train 10 --num_classes_test 10 --num_shots_train 10
--train_batch_size 100 --number_of_steps 80000 --num_tasks_per_batch 2
--num_classes_total 10 --num_samples_per_class 5000 --dataset cifar10
--model_name vgg --num_cases_train 50000 --num_cases_test 10000 --weight_decay
0.0005 --log_dir ./vgg11_cifar10/
```
