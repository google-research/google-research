# Clustering by learning to optimize expected Normalized Cuts (CNC)

The code for the experimentation made for the CNC.

## About the CNC

CNC is a framework for Clustering by learning to optimize expected Normalized Cuts. We show that by directly minimizing a continuous relaxation of the normalized cuts problem, CNC outperforms the traditional spectral clustering approach. Here is a motivational example that how CNC works.

Let us assume that we want to partition 6 images from the CIFAR-10 dataset (Figure 1) into two clusters. Given the affinity graph in this example, the optimal clustering is the result of cutting the edge connecting the two triangles. Cutting this edge, is exactly the the optimal solution for the normalized cuts. In CNC, we define a new differentiable loss function equivalent to the expected normalized cuts objective. We train a deep learning model to minimize the proposed loss in an unsupervised manner without the need for any labeled datasets. Our trained model directly returns the clustering probabilities. In this example, the optimal normalized cuts is 0.286, and as we can see, the CNC loss also converges to this value. Note that spectral clustering is also able to come up with the same cluster assignments by embedding the affinity of each pair of data points in Laplacian’s eigenspace and then uses k-means to generate clusters. The advantage of our work is in a new end to end differentiable method that directly minimizes a continuous relaxation of the normalized cuts.

![Motivation](https://drive.google.com/file/d/1qohoeNTNcJ9I0Zt1BInt-v2jspFiSvJK/view?usp=sharing)
Figure 1. Motivational Example.

## Understanding the code

For more detailed information, read the documentation within each file.

cnc_net.py: Contains run function for CNC.

networks.py: Contains network definitions for both CNC, Siamese Networks.

affinities.py: Contains all functions to construct affinity graph for CNC and Siamese Networks.

layer.py: Contains functions to build CNC, Siamese models.

train.py: Contains all training and prediction backend functions.


### Running the code

Run the ```run.py``` to train the CNC. All the hyperparameters with their description are defined in this file.

You can run the code for the mnist dataset by:
```python -m clustering_normalized_cuts.run --dset mnist```


