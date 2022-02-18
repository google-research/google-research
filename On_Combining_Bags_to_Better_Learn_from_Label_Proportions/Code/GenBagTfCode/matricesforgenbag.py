# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/python
#
# Copyright 2021 The On Combining Bags to Better Learn from
# Label Proportions Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Matrices for Bag Distn. Scenarios Method 1-5."""
import numpy as np
import scipy.stats

# For method 1
a = 0.2
b = 0.2
c = a + b - 2 * a * b

print("a, b, c: ", a, " ", b, " ", c)

Wmatrix = np.zeros((2, 2))

for i in range(1):
  Wmatrix[i][i] = b / (a * c)
  Wmatrix[i + 1][i + 1] = a / (b * c)
  Wmatrix[i][i + 1] = (-1) / c
  Wmatrix[i + 1][i] = (-1) / c

print("wmatrix:")
print(Wmatrix)

# Output for method 1
# [[ 3.125 -3.125]
# [-3.125  3.125]]

# For method 2 and 3

a = 0.9
b = 0.1
c = a + b - 2 * a * b

print("a, b, c: ", a, " ", b, " ", c)

Wmatrix = np.zeros((6, 6))

for i in range(3):
  Wmatrix[i][i] = b / (a * c)
  Wmatrix[i + 3][i + 3] = a / (b * c)
  Wmatrix[i][i + 3] = (-1) / c
  Wmatrix[i + 3][i] = (-1) / c

print("wmatrix:")
print(Wmatrix)


# Output for method 2 and 3
# [[ 0.13550136  0.          0.         -1.2195122   0.          0.        ]
# [ 0.          0.13550136  0.          0.         -1.2195122   0.        ]
# [ 0.          0.          0.13550136  0.          0.         -1.2195122 ]
# [-1.2195122   0.          0.         10.97560976  0.          0.        ]
# [ 0.         -1.2195122   0.          0.         10.97560976  0.        ]
# [ 0.          0.         -1.2195122   0.          0.         10.97560976]]

# For method 4
a, var, skew, kurt = scipy.stats.powerlaw.stats(1.66, moments="mvsk")
b = a
c = a + b - 2 * a * b

print("a, b, c: ", a, " ", b, " ", c)

Wmatrix = np.zeros((6, 6))

for i in range(3):
  Wmatrix[i][i] = b / (a * c)
  Wmatrix[i + 3][i + 3] = a / (b * c)
  Wmatrix[i][i + 3] = (-1) / c
  Wmatrix[i + 3][i] = (-1) / c

print("wmatrix:")
print(Wmatrix)

# Output for method 4
# [[ 2.13120482  0.          0.         -2.13120482  0.          0.        ]
# [ 0.          2.13120482  0.          0.         -2.13120482  0.        ]
# [ 0.          0.          2.13120482  0.          0.         -2.13120482]
# [-2.13120482  0.          0.          2.13120482  0.          0.        ]
# [ 0.         -2.13120482  0.          0.          2.13120482  0.        ]
# [ 0.          0.         -2.13120482  0.          0.          2.13120482]]

a = 0.2
b = 0.6
c = a + b - 2 * a * b

print("a, b, c: ", a, " ", b, " ", c)

Wmatrix = np.array([[b / (a * c), -1 / c, -1 / c, -1 / c],
                    [-1 / c, a / (b * c), a / (b * c), a / (b * c)],
                    [-1 / c, a / (b * c), a / (b * c), a / (b * c)],
                    [-1 / c, a / (b * c), a / (b * c), a / (b * c)]])

print("Wmatrix for method 5")
print(Wmatrix)

# Wmatrix for method 5
# [[ 5.35714286 -1.78571429 -1.78571429 -1.78571429]
#  [-1.78571429  0.5952381   0.5952381   0.5952381 ]
#  [-1.78571429  0.5952381   0.5952381   0.5952381 ]
#  [-1.78571429  0.5952381   0.5952381   0.5952381 ]]
