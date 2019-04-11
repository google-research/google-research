# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:07:32 2019

@author: Avinash Siravuru
"""

from hyperopt import fmin, tpe, hp
best = fmin(fn=lambda x: -x ** 2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=1000)
print(best)