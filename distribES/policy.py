import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import reverse_var_shape, flatvariable

class SimplePolicy:
    def __init__(self, obs_input=None, act_output=None, eps=0.1, clipped_act=1, 
                 clipped_obs=5, hidden=3, units=128, network=None, types='linear'):
        self.obs_input = obs_input
        self.act_output = act_output
        self.eps = eps
        self.clipped_act = clipped_act
        self.clipped_obs = clipped_obs
        self.hidden = hidden
        self.units = units
        self.types = types
        
        self.network = network if network is not None else self.initialize_net()
        
    def initialize_net(self):
        if self.types != 'linear':
            raise NotImplementedError
            
        ins = keras.layers.Input(self.obs_input)
        x = keras.layers.Dense(self.units, use_bias=True, activation='relu',)(ins)
        
        for i in range(self.hidden - 1):
            x = keras.layers.Dense(self.units, use_bias=True, activation='relu',)(x)
        outs = keras.layers.Dense(self.act_output)(x)
        
        return keras.Model(ins, outs)
    
    def forward_network(self, inputs):
        return self.network(inputs)
    
    def forward_policy(self, inputs):
        logit_k = self.network(inputs)
        if np.random.rand() <= self.eps:
            a = tf.squeeze(tf.random.categorical(logit_k, 1))
        else:
            a = tf.math.argmax(logit_k, 1)[0]
        return a
    
    def get_vars_flat(self):
        return flatvariable(self.network.get_weights()).numpy()
    
    def set_from_flat(self, var):
        self.network.set_weights(reverse_var_shape(var, self.network))
    
    def get_flattened_shape(self, full=True):
        if full:
            return self.get_vars_flat().shape
        else:
            raise NotImplementedError
    