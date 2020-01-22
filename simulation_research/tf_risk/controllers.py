# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Self-financing replicating portfolio controllers.

To replicate a sold derivative on some underlying asset two basic operations are
needed:
  1) Give an estimate, in the present market conditions, of the price of
the derivative. This determines how much initial cash is needed to initiate
the self-financing portfolio that replicates the derivative. The self-financing
portfolio is controlled to be equal in value to the derivative's payoff
at maturity.
  2) Give an estimate, in the present market conditions, of the hedging vector
of the derivative (usually called delta, it is typically the sensitivity of
the derivative's price w.r.t. the underlying asset price vector).

Both corresponding wrappers around tensorflow computation graphs are provided
here.
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from typing import Mapping, Union

NumpyArrayOrFloat = Union[np.ndarray, float]


def price_derivative(tf_price, tf_session,
                     params
                    ):
  """Wrapper executing the computation graph for a given derivative.

  Args:
    tf_price: a tensorflow tensor corresponding to the derivative's price
      estimate.
    tf_session: a tensorflow session which will execute the graph.
    params: a dictionary {placeholder: value} that will be fed into the session.

  Returns:
    A scalar estimating the derivative's price.
  """
  return tf_session.run(tf_price, feed_dict=params)


def hedge_derivative(tf_delta, tf_session,
                     params
                    ):
  """Wrapper executing the computation graph for a given derivative's delta.

  Args:
    tf_delta: a tensorflow tensor corresponding to the derivative's delta
      estimate.
    tf_session: a tensorflow session which will execute the graph.
    params: a dictionary {placeholder: value} that will be fed into the session.

  Returns:
    A numpy array estimating the derivative's delta.
  """
  return tf_session.run(tf_delta, feed_dict=params)
