# -*- coding: utf-8 -*-
""" Networks calculating transformations for discrete flows. """

import numpy as np
import tensorflow as tf  

import copy

import logging
logger = logging.getLogger(__name__)

import flows_edward2_made as made
#import edward2 as ed; made = ed.layers # uncomment to use original Edward2 implementation


class InputToOutputShift(tf.keras.layers.Layer):  
    """ Wraps a network (e.g., a RNN) so it could be used as a flow transformation.

        Ensures that output 0 does not depend on any input (only on logits0 param), 
        output 1 on input 0, output 2 on inputs no 0 and 1, etc.
    """       

    def __init__(self, rnn, logits0=None, **kwargs):
        super().__init__(**kwargs)        
        self.logits0 = logits0
        self.rnn = rnn

    def build(self, input_shape):
        N, K = input_shape[-2: ]
        if self.logits0 is None:
            self.logits0 = tf.Variable( tf.random.normal([K]), dtype=self.dtype, name="logits0")        
        super().build(input_shape)      
        logger.debug("[InputToOutputShift.build] name=%s network=%s" % (self.name, self.rnn.name))

    def call(self, x):        
        K = self.logits0.shape[-1]
        assert x.shape[-1]==K
        output0 = tf.broadcast_to(self.logits0, x.shape[:-2]+[1, K])
        output1 = self.rnn(x)[..., :-1, :]
        output = tf.concat([output0, output1], -2)
        return output
    

def build_rnn_model(vocab_size, rnn_units, 
                    embedding_dim=0, rnn_type=tf.keras.layers.GRU):

    rnn = tf.keras.Sequential(name="%s%i [K=%s, units=%s, embed=%s]" % 
                              (rnn_type.__name__, build_rnn_model._counter, 
                               vocab_size, rnn_units, embedding_dim))
    
    if embedding_dim>0:
        rnn.add( tf.keras.layers.Dense(embedding_dim) )            

    rnn.add( rnn_type(rnn_units,
             return_sequences=True,
             stateful=False,
             recurrent_initializer='glorot_uniform') )    
    rnn.add( tf.keras.layers.Dense(vocab_size) )        
     
    build_rnn_model._counter += 1
    return InputToOutputShift(rnn)
build_rnn_model._counter = 0


class ConditionedTransformation(tf.keras.layers.Layer):
    """ Wraps a transformation layer so it is conditional. 

        Conditioning is achieved by adding a condition 
        at the beginning of an input.
    """

    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)        
        self.layer = layer
        self._condition = None
        
    def call(self, input):
        condition = self.condition
        if condition is None:
            logger.warning("condition variable not set! initial calling: input.shape=%s" % input.shape)
            condition = input

        logger.debug("calling layer: input.shape=%s condition.shape=%s" % (input.shape, condition.shape))

        assert input.shape[:-2]==condition.shape[:-2] #batch equality
        assert input.shape[-1]==condition.shape[-1] #categories equality
        input_extended = tf.concat([condition, input], axis=-2)                
        output_extended = self.layer(input_extended)
        output = output_extended[..., condition.shape[-2]:, :]

        return output

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, value):
        self._condition = value
        logger.debug("[ConditionedTransformation] condition shape=%s" % self._condition.shape)


class CopiableMADE(made.MADE):

    def __init__(self,
               units,
               hidden_dims,
               input_order='left-to-right',
               hidden_order='left-to-right',
               activation=None,
               use_bias=True,
               **kwargs):
        super().__init__(
               units,
               hidden_dims,
               input_order=input_order,
               hidden_order=hidden_order,
               activation=activation,
               use_bias=use_bias, **kwargs)
        self._kwargs = copy.deepcopy(kwargs)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls(copy.deepcopy(self.units),
                     copy.deepcopy(self.hidden_dims),
                     copy.deepcopy(self.input_order),
                     copy.deepcopy(self.hidden_order),
                     copy.deepcopy(self.activation),
                     copy.deepcopy(self.use_bias),
                     **self._kwargs)
        result.network = tf.keras.Sequential(copy.deepcopy(self.network.layers))  #tf.keras.Sequential([])    
        result.built = True
        return result    

    def build(self, input_shape):
        super().build(input_shape)      
        #logger.debug("[CopiableMADE.build] name=%s shape=%s" % (self.name, input_shape))


class CopiableMADELocScale(CopiableMADE):

    def __call__(self, inputs, **kwargs):
        K = inputs.shape[-1]
        #the mask makes sure there are no 0 in scales by adding -inf on position of scale=0
        mask = tf.reshape([0] * K + [-1e10] + [0] * (K - 1), [1, 1, 2 * K])
        return mask + super().__call__(inputs, **kwargs)         



