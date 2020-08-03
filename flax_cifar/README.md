# Wideresnet and Pyramidnet on cifar10/cifar100

This is a FLAX re-implementation of the models used to reach state of the art on
cifar. This includes the implementation of the following:

-   Wideresnet ([Zagoruyko and Komodakis](https://arxiv.org/abs/1605.07146))
-   ShakeShake regularization ([Gastaldi](https://arxiv.org/abs/1705.07485))
-   Deep Pyramidal Residual Networks
    ([Han et al](https://arxiv.org/abs/1610.02915))
-   ShakeDrop Regularization ([Yamada et al](https://arxiv.org/abs/1802.02375))
-   AutoAugment policies ([Cubuk et al](https://arxiv.org/abs/1805.09501))

The implementation mimics the one from the autoaugment paper, which has becomed
a referenced for future works such as
[Lim et al](https://arxiv.org/abs/1905.00397) or
[Harris et al](https://arxiv.org/abs/2002.12047).

Using the same hyper-parameters as
[Cubuk et al](https://arxiv.org/abs/1805.09501), we obtain the following scores:

## Results with optimized parameters

When using several accelerators, is possible to increase the batch size in order
to decrease the wall clock time. The original paper used a batch size of 128 for
all model but pyramidnet, for which the batch size was 64. In our experiments,
we found that using a batch size of 256 (for all models) was a nice compromise
between generalization and training time. All parameters are kept the same as in
the reference, except for the learning rate that we gridsearch. To do so, we set
aside 10% of the training set that we use as a validation set, and select for
each model the learning rate that gave the best validation accuracy when trained
with autoaugment.

Scores below are computed on the test set, and are presented as **"our /
baseline"** where "our" is the mean error rate obtained with this codebase,
averaged over 5 runs. "baseline" are the scores reported in
[Cubuk et al](https://arxiv.org/abs/1805.09501).

#### Cifar10

Model                  | baseline | cutout  | AutoAugment
---------------------- | -------- | ------- | -----------
Wide-ResNet-28-10      | 3.7/3.9  | 2.8/3.1 | 2.4/2.6
Shake-Shake (26 2x96d) | 2.8/2.9  | 2.3/2.6 | 1.9/2.0
PyramidNet+ShakeDrop   | 2.4/2.7  | 1.8/2.3 | 1.6/1.5

#### Cifar100

Model                  | baseline  | cutout    | AutoAugment
---------------------- | --------- | --------- | -----------
Wide-ResNet-28-10      | 18.8/18.8 | 17.0/18.4 | 15.9/17.1
Shake-Shake (26 2x96d) | 17.0/17.1 | 15.7/16   | 14.1/14.3
PyramidNet+ShakeDrop   | 14.5/14   | 11.9/12.2 | 10.8/10.7

### Hyper parameters

The hyper-parameters used to reproduce the results above are:

Model                  | Learning rate | Weight decay | Epochs
---------------------- | ------------- | ------------ | ------
Wide-ResNet-28-10      | 0.1           | 0.0005*      | 200*
Shake-Shake (26 2x96d) | 0.02          | 0.001*       | 1800*
PyramidNet+ShakeDrop   | 0.1           | 0.0005       | 1800*

(* These values are taken from the reference implementation, we did not try to
tune them ourself.)

All models were trained with a batch size of 256, with max_grad_norm = 5.0 for
gradient clipping (as per the original implementation.)

## Result with reference hyper parameters.

For completeness, we also report the scores obtained using the hyper-parameter
provided in [Cubuk et al](https://arxiv.org/abs/1805.09501).

#### Cifar10

Model                  | baseline | cutout  | AutoAugment
---------------------- | -------- | ------- | -----------
Wide-ResNet-28-10      | 3.7/3.9  | 2.8/3.1 | 2.4/2.6
Shake-Shake (26 2x96d) | 2.7/2.9  | 2.2/2.6 | 1.9/2
PyramidNet+ShakeDrop   | 2.7/2.7  | 1.9/2.3 | 1.5/1.5

#### Cifar100

Model                  | baseline  | cutout    | AutoAugment
---------------------- | --------- | --------- | -----------
Wide-ResNet-28-10      | 18.2/18.8 | 16.3/18.4 | 15.6/17.1
Shake-Shake (26 2x96d) | 17.6/17.1 | 15.9/16   | 14.1/14.3
PyramidNet+ShakeDrop   | 16.0/14.0 | 13.3/12.2 | 11.3/10.7

Differences in the scores between this implementation and the original one stems
from the batch norm implementation. In the reference paper, the batch norm
statistics are computed on the whole batch. Our experiments were runned using 8
GPU, and batch norm statistics where computed independently for each replicas.
It is well known that, in addition to be more efficient, not synchronizing the
batch norm statistics is beneficial for generalization.

## Implementation details

Below are some implementation details that are needed to reach the expected
performance. Those are present in most papers, although not always made clearly
explicit.

-   Critically, we apply weight decay to the batch norm parameters. Not doing so
    results in noticably weaker performance.
-   The random mask for cutout regularization is the same for the whole batch
    (and not one mask per image). Mixup is not included in this version of the
    code, but if one wish to implement it, the same mixing coefficient must be
    used for all images in a batch.
-   This implementation of Pyramidnet rounds the number of filters instead of
    truncating it. This results in a model with roughtly 0.2M parameters more
    than reported in ([Han et al](https://arxiv.org/abs/1610.02915)). We however
    keep this behavior as most implementations of the Pyramidnet do so.
-   We use Kaiming with fan out initialization for consistency with
    ([Cubuk et al](https://arxiv.org/abs/1805.09501)), although we suspect that
    with batch norm this doesn't change much.

## Contact for issues:

Pierre Foret, pierreforet@google.com
