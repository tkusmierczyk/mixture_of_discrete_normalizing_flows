# -*- coding: utf-8 -*-
""" The most basic implementation of mixture of discrete flows 
    assuming factorized posterior and independence between variables. 
"""

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

import one_hot as utils


class FactorizedDiscreteFlowsMixture(tf.keras.layers.Layer):
    """ Mixture of discrete flows assuming factorized posterior and independence between variables. """
    
    def __init__(self, N, K, B, temperature=1.0, 
                 **kwargs): 
        super().__init__(**kwargs)        
        dtype = self.dtype

        probs = tf.one_hot(np.random.choice(np.arange(K), (N, B)), K, dtype=dtype) # delta probs
        self.components = tfd.OneHotCategorical(probs=probs, dtype=dtype) # N x B x K

        self.logits = tf.Variable( tf.random.normal((N, B, K), dtype=dtype), name="logits") 
        self.temperature = temperature           
        
    def sample_extm(self, n=1, *args, **kwargs):
        B = self.components.probs.shape[-2]
        dtype = self.dtype
        
        shift = utils.one_hot_argmax(self.logits, self.temperature) # N x B x K
        sample = self.components.sample(n) # n x N x B x K       
        sample = utils.one_hot_minus(sample, shift)
        
        selected_flows = (np.arange(n)+np.random.randint(B))%B # allocate equally between all flows
        mask = tf.one_hot(selected_flows, B, dtype=dtype)
        mask = mask[:,None,:,None] # n x 1 x B x 1
        sample = tf.reduce_sum(sample*mask, -2) #  n x N x K   
        return sample, mask

    def sample(self, n=1):
        return self.sample_extm(n)[0]
    
    def log_prob(self, sample, eps_prob=1e-31):
        B = self.components.probs.shape[-2]
        component_probs = self.components.probs
        
        shift = utils.one_hot_argmax(self.logits, self.temperature)
        sample = utils.one_hot_add(sample[:,:,None,:], shift[None,:,:,:])     
        
        prob = tf.reduce_sum(component_probs*sample + eps_prob, -1) # sum over categories => n x N x B
        log_prob = tf.math.log(prob) + np.log(1./B) # n x N x B
        log_prob = tf.math.reduce_logsumexp(log_prob, -1) # sum over B mixture components  => n x N
        return tf.reduce_sum(log_prob, -1) # sum over N

    def call(self, sample, *args, **kwargs): # Legacy compatible
        return sample

    def reverse(self, sample, *args, **kwargs): # Legacy compatible
        return sample

    def log_prob_ext(self, sample, *args, **kwargs): # Legacy compatible
        return self.log_prob(sample)

