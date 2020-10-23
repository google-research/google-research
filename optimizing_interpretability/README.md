# Directory Structure

This directory contains code to train models with input gradient regularization on ImageNet.

## `regularizers.py`

This script is imported by all model subfolders to compute the regularizer loss.
These regularizer losses are based upon metrics from literature on image
processing and the human visual system. Losses implemented here are described
below.

#### 1) Standalone regularizers:

-   **L2:** standard double backpropagation, as a reference. We try the original
    formulation (gradient of the loss), the OneHot formulation (gradient of the
    logit for the true class), and the SpectReg formulation (gradient of a
    random projection of the logits’ Jacobian).

-   **Total variation:** an alternative penalty imposing a spatial smoothing
    constraint on the heatmap. We evaluate both total variation (using the raw
    gradients) and absolute total variation (evaluated on the absolute value of
    the gradients).

#### 2) Reference-point regularizers:

-   **Mean squared error (MSE)**: penalizes the pixel-wise squared difference
    between the image and the heatmap, after normalizing both to the same
    dynamic range. We also try maximizing the correlation coefficient between
    the two.

-   **Gradient difference**: penalizes the squared difference between the
    horizontal and vertical image gradients for the input and the heatmap,
    rather than individual pixels.

-   **Sobel edge loss**: penalizes the squared difference between Sobel edge
    maps for the input and the heatmap.

-   **Peak signal-to-noise ratio (PSNR)**: a measurement of the fidelity of the
    heatmap to the original image, proportional to log-MSE. PSNR goes to
    infinity as MSE goes to zero, so we minimize its reciprocal. We also use a
    modified version of PSNR, called PSNR-HVS (as proposed by Gupta et al.,
    2011), that accounts for measures of edge and structural distortion.

-   **Structural similarity index (SSIM)**: a metric computed over patches of
    the input and heatmap, comparing features such as luminance, contrast, and
    structure. We try two versions with two different filters for computing over
    patches (one Gaussian, one moving average).

## Training pipelines

### ImageNet

Scripts for optimizing checkpoints using perceptual metrics. Includes the following:

-   `preprocessing_helper.py`: preprocessing transformations of test and train images.

-   `data_helper.py`: utils for loading and preprocessing ImageNet.

-   `resnet_model.py`: ResNet definition.

-   `resnet_train.py`: Training script to finetune checkpoint.

-   `utils.py`: additional utils to add noise to image inputs and explanations.
