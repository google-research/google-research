# Generating unfoldings of the hypercube

We provide an open source C++ implementation which can be used to generate all
unfoldings of the n-dimensional hypercube. Those unfoldings correspond to pairs
(T, P), where T is a tree on 2n nodes and P is a perfect matching in the
complement of the tree. Two such pairs (T, P) and (T', P') are considered
equivalent if there is an isomorphism between T and T' which is simultaneously
an isomorphism between P and P'.

This correspondence is explained in
[*Unfolding the Tesseract*](https://unfolding.apperceptual.com/) by Peter
Turney.

We rely on [nauty](https://pallini.di.uniroma1.it/) by Brendan McKay and Adolfo
Piperno to generate the trees and their automorphism group.

The code is used to calculate the terms [A091159](https://oeis.org/A091159) up
to dimension 10, see below for the results.

Check out a [video by Matt Parker on unfolding the 4d-cube](https://www.youtube.com/watch?v=Yq3P-LhlcQo).


# Compiling
Use `./compile.sh` to compile the source code. You can then use `./test.sh` to
run tests, or `./compute.sh N` to run the computation for a given value of `N`.

Compiling the code requires `nauty` and `gtest`. Running requires the
`nauty-gtreeng` executable, as well as `parallel`.

To install these dependencies on a Debian-based system, you can run

```
sudo apt-get install libnauty2-dev nauty libgtest-dev parallel
```

# Results

For each n ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10} we provide a file `n.cnt.txt` which
contains in each line a decimal number and a graph given in [sparse6
format](https://users.cecs.anu.edu.au/~bdm/data/formats.txt) separated by a
space.

```
[number] [graph6 string]
```

For example

```
11704 :M`ESYOl]sLZt
```

The graph is a tree on 2*n vertices and the number counts the perfect matchings
in the complement of the tree up to transformations of the automorphism group of
the tree. In other words the number tells us how many different unfoldings of
the hypercube in dimension n exists, which correspond to a certain tree. Some
trees in the files don't have any corresponding unfoldings, namely the stars.
They are nonetheless listed here with a count of zero.

To get the number of hypercube unfoldings we can add up all the numbers.

dimension | number of unfoldings | file
--------: | -------------------: | :---
2         | 1                    |[2.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/2.cnt.txt)
3         | 11                   |[3.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/3.cnt.txt)
4         | 261                  |[4.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/4.cnt.txt)
5         | 9694                 |[5.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/5.cnt.txt)
6         | 502110               |[6.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/6.cnt.txt)
7         | 33064966             |[7.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/7.cnt.txt)
8         | 2642657228           |[8.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/8.cnt.txt)
9         | 248639631948         |[9.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/9.cnt.txt)
10        | 26941775019280       |[10.cnt.txt](http://storage.googleapis.com/gresearch/cube_unfoldings/results/10.cnt.txt)


Authors: [Moritz Firsching](https://mo271.github.io/) and [Luca
Versari](https://lucaversari.it).
