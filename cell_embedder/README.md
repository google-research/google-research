# Generating embeddings for cell painting microscopy images.

This python notebook generates embeddings for microscopy images stained
using cell painting assays.

This was used in papers:

1. [Applying Deep Neural Network Analysis to High-Content Image-Based Assays. (in SLAS Discovery)][slas_paper]
2. [It's easy to fool yourself: Case studies on identifying bias and confounding in bio-medical datasets. (in NeurIPS 2019 workshops)][neurips_lmrl_paper]

## Download the sample data and weights.

To run the accompanying colab, you need:

1. A sample image (5 stains), and
2. weights to the random projection matrix.

These files are available for [download here][colab_downloads]. We use sample
images from [BBBC025][bbbc025_link] in this notebook.

Other data accompanying the [paper][slas_paper] are [here][paper_downloads]


## Running the notebook

1. Run the cells in **Install + Imports**.

2. [Download the weights and sample images here][colab_downloads]. You should be
   able to select "Add shortcut to drive" to run this notebook. Note that you
   can use your own images, but you'll likely need the weights to get an
   embedding identical to the ones in the paper.

3. Set the `DATA_DIR` variable to the location where the images and weights
   live, and make sure to run that cell to initialize the paths to the images
   and the model weights.

4. Run the subsequent cells: **Helper functions** and **Load Images (sorted by
   stain names)**. You should be able to see the sample images. Also run
   **Build model and initialize weights**.

5. Now run the cell under **Get Embeddings** section. It should give you a 320d
   vector and will plot it.

## Resources/Links

1. [Applying Deep Neural Network Analysis to High-Content Image-Based Assays. (in SLAS Discovery)][slas_paper]
2. [It's easy to fool yourself: Case studies on identifying bias and confounding in bio-medical datasets. (in NeurIPS 2019 workshops)][neurips_lmrl_paper]
3. [Link to download weights and sample image to run the
   notebook][colab_downloads]
4. (TO COME SOON) [Link to download full data accompanying the paper][paper_downloads]



This project is not an official Google product.

[slas_paper]: https://journals.sagepub.com/doi/full/10.1177/2472555219857715
[neurips_lmrl_paper]: https://arxiv.org/abs/1912.07661
[bbbc025_link]: https://bbbc.broadinstitute.org/BBBC025
[colab_downloads]: https://drive.google.com/drive/folders/1dTE0PQTMmbg-H0nMjNSPGBfRqinIfFud?usp=sharing
[paper_downloads]: https://storage.googleapis.com/project_name/all_data.zip



