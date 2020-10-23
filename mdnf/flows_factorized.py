# -*- coding: utf-8 -*-
""" Individual discrete flows for factorized distributions. """


import numpy as np
import tensorflow as tf

import one_hot as utils
#import edward2; utils = edward2.layers.utils # to use original Edward2's operations

import logging
logger = logging.getLogger(__name__)


class DiscreteFactorizedFlow(tf.keras.layers.Layer):     
    """ Discrete (shift-only) flow assuming dimensions 0..N-1 to be independent. """

    def __init__(self, N, K, temperature=1.0, 
                 logits=None, **kwargs):
        super().__init__(**kwargs)
        
        if logits is None: 
            logits = tf.Variable( tf.random.normal((N,K), dtype=self.dtype), 
                                    dtype=self.dtype, name="logits")
        assert logits.shape==(N,K), "expected logits shape=(N=%s, K=%s)" % (N,K)
        self._logits = logits
        self._temperature = temperature

    @staticmethod
    def call_static(x, logits, temperature):
        shift = utils.one_hot_argmax(logits, temperature)
        return utils.one_hot_minus(x, shift)
     
    @staticmethod
    def reverse_static(x, logits, temperature):
        shift = utils.one_hot_argmax(logits, temperature)
        return utils.one_hot_add(x, shift)

    def call(self, x):
        return self.call_static(x, self.logits, self.temperature)

    def reverse(self, x):
        return self.reverse_static(x, self.logits, self.temperature)

    @property
    def trainable_variables(self):
        return [self._logits]
        
    def select_trainable_variables(self, *args, **kwargs):
        return self.trainable_variables

    @property
    def temperature(self):
        return self._temperature          

    def set_temperature(self, t):
        try:     self._temperature.assign(t)
        except:  self._temperature = t

    @property
    def logits(self):        
        return self._logits

    @logits.setter
    def logits(self, value):
        self._logits = value

    @temperature.setter
    def temperature(self, t):
        self.set_temperature(t)

    @property
    def K(self):
        return self.logits.shape[-1]


class DiscreteFactorizedFlowLocScale(DiscreteFactorizedFlow):     
    """ Discrete location-scale flow assuming dimensions 0..N-1 to be independent. """

    def __init__(self, N, K, temperature=1.0, 
                 logits=None, logits_scale=None, **kwargs):
        super().__init__(N, K, temperature, logits, **kwargs)
        if logits_scale is None: 
            logits_scale = tf.Variable( tf.random.normal((N,K), dtype=self.dtype), 
                                            dtype=self.dtype, name="logits_scale")
        self._logits_scale = logits_scale
        assert logits_scale.shape == (N,K)
        self._logits_scale_mask = tf.concat([tf.ones((N,1))*-1e30, tf.zeros((N,K-1))], 1) # prevents from scale=0

    def call(self, x):
        shift = utils.one_hot_argmax(self.logits, self.temperature)
        shifted_inputs = utils.one_hot_minus(x, shift)
        scale = utils.one_hot_argmax(self.logits_scale, self.temperature)
        inverse_scale = utils.multiplicative_inverse(scale, self.K)
        return utils.one_hot_multiply(shifted_inputs, inverse_scale)

    def reverse(self, x):
        scale = utils.one_hot_argmax(self.logits_scale, self.temperature)
        scaled_inputs = utils.one_hot_multiply(x, scale)
        shift = utils.one_hot_argmax(self.logits, self.temperature)       
        return utils.one_hot_add(shift, scaled_inputs)

    @property
    def trainable_variables(self):
        return [self._logits, self._logits_scale]
        
    def select_trainable_variables(self, *args, **kwargs):
        return self.trainable_variables

    @property
    def logits_scale(self):        
        return self._logits_scale + self._logits_scale_mask

    @logits_scale.setter
    def logits_scale(self, value):
        self._logits_scale = value



class DiscreteFactorizedFlowPartial(DiscreteFactorizedFlow):
    """ Discrete shift-only flow acting on subset of categories. """

    def __init__(self, N, K, categories, **kwargs):
        """
            Args:
                categories  A list of category numbers to be transformed.
        """
        subset_K = len(categories)
        assert subset_K <= K
        super().__init__(N, subset_K, **kwargs)

        shuffling = np.arange(K)
        np.random.shuffle(shuffling)
        shuffling = np.array(list(categories)+[v for v in shuffling if v not in categories])

        inverted_shuffling = np.arange(K)
        inverted_shuffling[shuffling] = np.arange(K)

        self.shuffling = shuffling
        self.inverted_shuffling = inverted_shuffling
        self.subset_K = subset_K

    def call(self, x):
        """Applies flow to first self.subset_K shuffled categories."""
        x = tf.gather(x, self.shuffling, axis=-1)
        x1 = x[..., : self.subset_K]
        x1 = super().call(x1)
        x2 = x[...,self.subset_K: ]
        x = tf.concat([x1,x2], axis=-1)
        x = tf.gather(x, self.inverted_shuffling, axis=-1)
        return x

    def reverse(self, x):
        """Applies reverse flow to first self.subset_K shuffled categories."""
        x = tf.gather(x, self.shuffling, axis=-1)
        x1 = x[..., : self.subset_K]
        x1 = super().reverse(x1)
        x2 = x[...,self.subset_K: ]
        x = tf.concat([x1,x2], axis=-1)
        x = tf.gather(x, self.inverted_shuffling, axis=-1)
        return x


