# Milking CowMask for semi-supervised image classification

Intern project by Geoff French, hosted by Avital Oliver and Tim Salimans

This project explores the use of CowMask for semi-supervised image classification.
It achieves competitive results on CIFAR-10 using a 26 layer Wide ResNet with Shake-shake regularization.
It also achieves competitive results on ImageNet.

To train a network on CIFAR-10 using 1,000 supervised samples:

> python main_semisup.py

The default values for the command line arguments should replicate a single result from our upcoming paper.