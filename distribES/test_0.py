import tensorflow as tf
from tensorflow import keras
from atari_preprocessing import  AtariPreprocessing
import gym
from policy import SimplePolicy
from es import *

def create_simple_conv(input_dim, output_dim, actv='relu', output_actv='linear', seed_i=212, name='conv'):
    inits = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=seed_i) 

    ins = keras.layers.Input(shape=input_dim)
    x = keras.layers.Conv2D(filters=64, kernel_size=(6, 6), strides=(2, 2), activation=actv, use_bias=True,
            kernel_initializer=inits, bias_initializer='zeros')(ins)
    x = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation=actv, use_bias=True,
            kernel_initializer=inits, bias_initializer='zeros')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation=actv, use_bias=True,
            kernel_initializer=inits, bias_initializer='zeros')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(300, activation=actv)(x)
    out = keras.layers.Dense(output_dim, activation=output_actv)(x)

    return keras.Model(ins, out)


env_name = "ALE/Boxing-v5"

latent_dim = (80, 80, 1)
act_dim = 18

net = create_simple_conv(latent_dim, act_dim, actv='relu', output_actv='linear')
max_epoch = 50
max_generation = 100

policy0 = SimplePolicy(network=net, eps=0)

manager0 = DESmanager(env_name, latent_dim, act_dim, n_workers=100, max_gen=max_generation, max_steps=1000, 
                      policy=policy0, std=0.01, log_dir=None, lr=0.0001, seed=15)

summary = tf.summary.create_file_writer(logdir='Evo')
with summary.as_default():
    manager0.train(max_epoch, save_each=False)
    
manager0.policy.network.save(env_name)