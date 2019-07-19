# A Simple Dictionary Learning Implementation

The basic dictionary learning is stated as follows. Given a matrix A of size m x n (in the context of machine learning n is the number of features and m is the number of observations), we want to approximate the original matrix A by a product of a “sparse” representation C and a dictionary D. In this work, we seek to minimize the Frobenius norm of E = A - CD. See https://en.wikipedia.org/wiki/Sparse_dictionary_learning for more details.

In the factorization, the representation C is a sparse matrix of size m x m_D and the dictionary D has size m_D x n, where m_D << m and each row of C has at most t non-zero entries. Intuitively, each row of A is approximated by a linear combination of t rows of the dictionary D. Note that this problem is known to be NP-hard.

We define the compression ratio r := (mt + m_Dn) / (mn). Normally, we will pick t and m_D such that r is in the range [0.1, 0.25].

In this work, we study a simple iterative method. First, we initialize either C by kNN (k nearest neighbors). (Note that, for initialization of C and D, we can just start with random matrices. However, the algorithm converges much more quickly if we guess a better initial value of C or D). Next, we alternatively optimize C and D in each iteration:
* compute the best-possible representation C based on the current dictionary D -- this problem is significantly easier as D is fixed,
* compute the best-possible dictionary D based on the current representation C -- this is simply a least-squares problem.

As mentioned above, we will alternatively solve for C (or D) given a fixed value of D (or C) in each iteration:
* Optimizing the representation C: This problem is known as the Orthogonal Matching Pursuit (OMP) problem:
argmin_C || A - CD ||^2 subject to a constraint that each row of C has at most t non-zero entries. (This problem can be solved in polynomial time.)

Authors: Khoa Trinh (corresponding author -- email: khoatrinh@google.com), Rina Panigrahy‎, and Badih Ghazi‎.
￼
As long as it is somewhere
maybe put authors in the pdf file too in that same order
Put email address after the three authors

* Optimizing the dictionary D: Again, we want to find D such that ||A-CD||^2 is minimized. Observe that this is just a least-squares problem.

The main function is 'dictionary_learning' in dictionary_learning.py which takes input as a matrix and parameters to determine the compression ratio
and returns 'code' and 'dictionary' such that the original matrix can be approximated by 'code' * 'dictionary'. See its documentation for more details.

Authors: Khoa Trinh (corresponding author -- email: khoatrinh@google.com), Rina Panigrahy‎, and Badih Ghazi‎.
