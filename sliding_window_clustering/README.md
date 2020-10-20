Sliding Window Algorithms for k-Clustering Problems
===============================

This is the implementation accompanying the NeurIPS 2020 paper,
[_Sliding Window Algorithms for k-Clustering Problems_](https://arxiv.org/abs/2006.05850).


Example Usage:
--------------

The main program takes in input a text file where each line represents a point
in the Euclidean space of d dimension. Each line consists of d floating point
fields separated by tab character '\t'. The output is a text file where each
line corresponds to a computation of a solution over the sliding window and
reports:
1) the update time of our algorithm (i.e. number of distance function
computations in the update procedure);
2) the memory of our algorithm as the number of items stored;
3) the k-means cost of the solution of our algorithm;
4) the k-means cost of the gold standard baseline of running k-means++
 over the entire sliding window;
5) the k-means cost of a baseline solution using sampling at parity of items
 stored by our algorithm.


How to run:
-------
Our code uses *bazel* to build the C++ binary.

You can run our experiments with the following:

1) First download the repository and install bazel if needed.

2) Run the code.

cd sliding_window_clustering/

bazel run -c opt :sliding_window_clustering -- \
 --input_file="/path/to/input.txt" \
 --output_file="/path/to/output-input.txt" \
 --window_size=10000 \
 --k=10

To turn on the debug output use '-c dbg' instead (notice that this will make the
code significantly slower).

Citing
------
If you find _Sliding Window Algorithms for k-Clustering Problems_ useful in your
research, we ask that you cite the following paper:

> Borassi, M., Epasto, A., Lattanzi, S., Vassilvitskii, S., Zadimoghaddam, M. (2020).
> Sliding Window Algorithms for k-Clustering Problems
> In _NeuriPS_.

    @inproceedings{borassi2020sliding,
     author={Borassi, Epasto, Lattanzi, Vassilvitskii, Zadimoghaddam}
     title={Sliding Window Algorithms for k-Clustering Problems},
     booktitle = {NeurIPS},
     year = {2020},
    }

Contact Us
----------
For questions or comments about the implementation, please contact
<aepasto@google.com>.
