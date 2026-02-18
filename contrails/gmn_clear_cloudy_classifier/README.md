# Clear/Cloudy Classifier for GMN

This package contains the code, model weights, and training data generation
scripts for a "Clear vs. Cloudy" classification task on Global Meteor Network
(GMN) ground camera imagery.

TLDR: A resnet18 classifier, fine-tuned on style-transferred (night -> day by
Gemini Nano Banana) GMN imagery, yields a clear/cloudy classifier that
generalizes well to real GMN daytime imagery. The model can then be used for
filtering observations down to times when contrail observations are not expected
to be obscured by natural cloud cover. Night -> day style transferred images are
used because "clear/cloudy" labels are already available during the nighttime
for GMN imagery using visibility of stars as the indicator for clear sky.

## Documentation

The documentation has been split into two parts based on use case:

*   **[Running Inference](README_inference.md)**: Instructions for using the
    trained model to classify images.
*   **[Training & Data Reproduction](README_training.md)**: Instructions for
    reproducing the training dataset from scratch, training the model, and
    selecting an operating point.

## Quick Start

1.  **Install Dependencies:** `bash virtualenv venv source venv/bin/activate pip
    install --index-url https://pypi.org/simple -r requirements.txt`

2.  **Run Inference:** `bash python inference.py`

For detailed instructions, please refer to the specific README files linked
above.
