import tensorflow as tf 
import numpy as np
import ray

def flatvariable(var):
    return tf.concat(axis=0, values=[tf.reshape(v, [tf.reduce_prod(v.shape)]) for v in var])

def reverse_var_shape(var_flatten, net0):
    var_flatten = np.array(var_flatten)
    v = []
    prods = 0
    for each_layer in net0.get_weights():
        shape= each_layer.shape
        prods0 = int(prods + np.prod(shape))
        v.append(var_flatten[prods:prods0].reshape(shape))
        prods = prods0
    return v 

@ray.remote
def create_shared_noise():
    count = 10_000_000
    noise = np.random.RandomState(3535).randn(count).astype(np.float64)
    return noise

class SharedNoiseTable(object):
    def __init__(self, noise, seed = 11):
        self.rg = np.random.RandomState(seed)
        self.noise = noise 
    def get_noise(self, i, dim):
        return self.noise[i:i + dim]
    def sample_index(self, dim):
        return self.rg.randint(0, len(self.noise) - dim + 1)
    def get_idx(self, dim): # call first
        idx = self.sample_index(dim)
        return idx, self.get_noise(idx, dim)