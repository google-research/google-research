import numpy as np
import scipy as sp

RAD_TO_DEG = 57.2957795
DEG_TO_RAD = 1.0 / RAD_TO_DEG
INVALID_IDX = -1

def lerp(x, y, t):
    return (1 - t) * x + t * y

def log_lerp(x, y, t):
    return np.exp(lerp(np.log(x), np.log(y), t))

def flatten(arr_list):
    return np.concatenate([np.reshape(a, [-1]) for a in arr_list], axis=0)

def flip_coin(p):
    rand_num = np.random.binomial(1, p, 1)
    return rand_num[0] == 1


        
    
