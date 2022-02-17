# TriMap: Large-scale Dimensionality Reduction Using Triplets

A JAX implementation of TriMap: examples using the current implementation will
be added soon.

TriMap is a dimensionality reduction method that uses triplet constraints to
form a low-dimensional embedding of a set of points. The triplet constraints are
of the form "point i is closer to point j than point k". The triplets are
sampled from the high-dimensional representation of the points and a weighting
scheme is used to reflect the importance of each triplet.

TriMap provides a significantly better global view of the data than the other
dimensionality reduction methods such t-SNE, LargeVis, and UMAP. The global
structure includes relative distances of the clusters, multiple scales in the
data, and the existence of possible outliers.

**Example usage:**

    import jax.random as random
    from sklearn.datasets import load_digits
    import trimap

    digits = load_digits()
    key = random.PRNGKey(42)

    embedding = trimap.transform(key, digits.data, distance='euclidean')

**Source:**

Paper: https://arxiv.org/pdf/1910.00204.pdf \
Github: https://github.com/eamid/trimap

**Reference:**

@article{2019TRIMAP, author = {{Amid}, Ehsan and {Warmuth}, Manfred K.}, title =
"{TriMap: Large-scale Dimensionality Reduction Using Triplets}", journal =
{arXiv preprint arXiv:1910.00204}, archivePrefix = "arXiv", eprint =
{1910.00204}, year = 2019, }
