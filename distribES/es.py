import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import gym
import scipy
from scipy import stats
import ray

from policy import SimplePolicy
from utils import *
from atari_preprocessing import  AtariPreprocessing

global_steps = tf.Variable(0, dtype=tf.int64, name='global_steps')

@ray.remote
class Worker:
    def __init__(self, env_name, sampled_noise=None, std=0.005, env_seed=555, seed=555,
                 policy=None, max_steps=3000, state_prep_fn=None):
        # note that seed need to be fix and same as in managerES
        self.env_name = env_name
        self.env = gym.make(self.env_name, frameskip=1)
        self.env = AtariPreprocessing(self.env, noop_max= 30,
            frame_skip= 4,
            screen_size= 80,
            terminal_on_life_loss= True,
            grayscale_obs= True,
            grayscale_newaxis= True,
            scale_obs= True)
        
        self.env_seed = env_seed
        self.policy = policy
        self.state_prep_fn = state_prep_fn
        self.max_steps = max_steps
        self.noise = SharedNoiseTable(sampled_noise, seed)
        self.std = std
        
        try:
            self.env.seed = env_seed
        except:
            ValueError('env has no attribute "seed"')
            
    def one_rollout(self):
        Rews, Stps = 0, 0
        S = self.env.reset()
        for t in range(self.max_steps - 1):
            S = tf.expand_dims(S, 0)
            A = self.policy.forward_policy(S).numpy()
            S, R, is_done, _ = self.env.step(A)
            Rews += R
            Stps += 1
            if is_done:
                break
        return Rews, Stps
    
    def purturb(self, var_policy, n_rollout=1): # TODO: implement partial weight purturbations
        assert (var_policy.shape == self.policy.get_flattened_shape())
        idxs, Batch_R, Stps = [], [], 0
        for k in range(n_rollout):
            idx, noises = self.noise.get_idx(self.policy.get_flattened_shape()[0])
            
            self.policy.set_from_flat(var_policy + self.std * noises)
            r_pos, s_pos = self.one_rollout()
            
            self.policy.set_from_flat(var_policy + self.std * - noises) 
            r_neg, s_neg = self.one_rollout()

            Stps += (s_pos + s_neg)
            Batch_R.append([r_pos, r_neg])
            idxs.append(idx)
        Batch_R = tf.convert_to_tensor(Batch_R, dtype=tf.float32)
        return Batch_R, idxs, Stps

class DESmanager:
    """Worker manager for distributed learning"""
    def __init__(self, env, obs, acts, n_workers=2, max_gen=100, max_steps=3000, policy=None ,
                 std=0.005, log_dir=None, lr=0.01, seed=555, env_seed=555):
        if n_workers>=1:
            #check n of available CPU/GPU, warn if less than n_workers
            pass
        self.n_workers = n_workers
        self.max_gen = max_gen
        self.max_steps = max_steps
        self.workers = None
        self.env = env # is env name not env
        self.obs_input = obs
        self.act_input = acts
        self.std = std
        self.log_dir = log_dir if log_dir is not None else 'RunResultES'
        self.lr = lr
        self.seed = seed
        self.env_seed = env_seed
        self._best_score = -np.inf
        self.policy = policy # note: this policy will always change its weights in sampling phase
        self.parameters = self.policy.get_vars_flat()
        self.state_prep_fn = None
        noise_sync_id = create_shared_noise.remote() 
        self.noise = SharedNoiseTable(ray.get(noise_sync_id), self.seed)
        self.workers = [Worker.remote(env_name=self.env, 
                                      sampled_noise=noise_sync_id, 
                                      std=self.std, 
                                      seed=self.seed + 1 * i_w, ##try varying seed
                                      env_seed=self.env_seed + 2 * i_w, 
                                      policy=self.policy,
                                      max_steps=self.max_steps,
                                      state_prep_fn=self.state_prep_fn) for i_w in range(self.n_workers)]
        self.optimizer = None
        
    def sync_returns(self, n_rollout=3):
        #self.max_gen = None set to 2 * step/workers
        var_policy_id = ray.put(self.parameters)
        
        Result_raw_obj = [wolker.purturb.remote(var_policy_id, n_rollout=n_rollout) for wolker in self.workers]
        Result = ray.get(Result_raw_obj)

        batch_r = tf.convert_to_tensor([res[0] for res in Result], dtype=tf.float32) # TODO: use namedtuple
        batch_idx = tf.convert_to_tensor([res[1] for res in Result], dtype=tf.int32)
        mean_steps = int(tf.reduce_sum([res[2] for res in Result]) / (2 * n_rollout * self.n_workers))
        global_steps.assign_add(len(batch_r))
        return batch_r, batch_idx, mean_steps
    
    def train_step(self, n_rollout=3, batch_sum=False):
    
        batch_r, batch_idx, mean_steps = self.sync_returns(n_rollout=n_rollout)
        Ns_ = np.prod(batch_r.shape)
        batch_r = tf.reshape(batch_r, (np.prod(batch_r.shape[:-1]), 2))
        mean_r = tf.reduce_mean(tf.reshape(batch_r, -1))
        batch_r = self.compute_centered_ranks(batch_r.numpy())
        batch_idx = tf.reshape(batch_idx, (np.prod(batch_idx.shape), -1))

        rec_noise = tf.convert_to_tensor(
            [self.noise.get_noise(ids[0].numpy(), self.policy.get_flattened_shape()[0]) for ids in batch_idx]
        , dtype=tf.float32)
        
        if not batch_sum:
            batch_r = tf.expand_dims(batch_r[:, 0] - batch_r[:, 1], -1)
            sums = tf.reduce_sum((batch_r * rec_noise), axis=0)
        else:
            raise NotImplementedError
        g = (1 / (Ns_ * self.std)) * sums
        if self.optimizer is None:
            self.parameters = self.parameters + self.lr * g
            self.policy.set_from_flat(self.parameters)
        else:
            raise NotImplementedError('not currently support update though optimizer')
        
        norm_upt = np.linalg.norm(self.parameters)
        return mean_r, norm_upt, mean_steps
    
    def train(self, epochs=5, save_each=True):
        for ep in range(epochs):
            rolls = int(self.max_gen / self.n_workers)
            mean_r, norm_update, mean_steps = self.train_step(rolls)
            print('mean_reward : ', mean_r.numpy(), 'norm_update : ', norm_update, 'mean_steps : ', 
                  mean_steps, 'env_steps : ', global_steps.numpy())
            tf.summary.scalar('mean_reward', mean_r, ep)
            tf.summary.scalar('norm_update', norm_update, ep)
            tf.summary.scalar('mean_steps', mean_steps, ep)
            if save_each:    
                self.policy.network.save('SavedModel/{}/{}'.format(self.env, str(ep)))
        self.policy.set_from_flat(self.parameters)
        
    def compute_centered_ranks(self, x):
        y = (stats.rankdata(x.ravel()) - 1).reshape(x.shape).astype(np.float32)
        return y / (x.size - 1) - 0.5
