# Numpy <-> nda::array conversion utilities

This is not an official Google product.

## Overview

Exposes `npy_array::SerializeToString` to serialize `nda::array` (from the
[array](https://github.com/dsharlet/array) library) instances to the numpy
array format (https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).

## Code structure

* /npy_array : C++ library
* /tests : Tests for the C++ library

------

### License

Apache 2.0.
